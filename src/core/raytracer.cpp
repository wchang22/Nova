#include "raytracer.hpp"
#include "constants.hpp"
#include "model/model.hpp"
#include "scene/scene.hpp"
#include "util/profiling/profiling.hpp"
#include "vector/vector_conversions.hpp"

namespace nova {

Raytracer::Raytracer() {
  accelerator.add_kernel("kernel_raytrace");
  accelerator.add_kernel("kernel_interpolate");
  accelerator.add_kernel("kernel_fill_remaining");
  accelerator.add_kernel("kernel_post_process");
}

void Raytracer::set_scene(const Scene& scene) {
  PROFILE_SCOPE("Set Scene");

  const uint32_t width = static_cast<uint32_t>(scene.get_output_dimensions()[0]);
  const uint32_t height = static_cast<uint32_t>(scene.get_output_dimensions()[1]);

  // Update Scene Params
  const vec3f& shading_diffuse = scene.get_shading_diffuse();
  const float shading_metallic = scene.get_shading_metallic();
  const float shading_roughness = scene.get_shading_roughness();
  const float shading_ambient_occlusion = scene.get_shading_ambient_occlusion();
  const vec3f& light_position = scene.get_light_position();
  const vec3f& light_intensity = scene.get_light_intensity();
  const int ray_bounces = scene.get_ray_bounces();
  const float exposure = scene.get_exposure();
  const bool anti_aliasing = scene.get_anti_aliasing();

  scene_params_wrapper = accelerator.create_wrapper<SceneParams>(SceneParams {
    scene.get_camera_eye_coords(), vec_to_float3(light_position), vec_to_float3(light_intensity),
    vec_to_float3(shading_diffuse), shading_metallic, shading_roughness, shading_ambient_occlusion,
    ray_bounces, exposure, anti_aliasing });

  // Update buffers depending on width, height
  if (this->width != width || this->height != height) {
    pixel_im = accelerator.create_image2D_write<uchar4>(ImageChannelOrder::RGBA,
                                                        ImageChannelType::UINT8, width, height);
    // Use a packed uchar4 image to save memory and bandwidth
    temp_pixel_im1 = accelerator.create_image2D_readwrite<uchar4>(
      ImageChannelOrder::RGBA, ImageChannelType::UINT8, AddressMode::CLAMP, FilterMode::NEAREST,
      false, width, std::max(height / 2, 1U));
    temp_pixel_im2 = accelerator.create_image2D_readwrite<float4>(
      ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
      true, width, height);

    pixel_dims_wrapper = accelerator.create_wrapper<uint2>(uint2 { width, height });
    rem_coords_buf =
      accelerator.create_buffer<int2>(MemFlags::READ_WRITE, std::max(width * height / 2, 1U));
  }

  // Update Model
  const std::string& model_path = scene.get_model_path();
  if (model_path != loaded_model) {
    intersectable_manager.clear();
    material_loader.clear();

    Model model(model_path, material_loader);
    intersectable_manager.add_model(model);

    auto [triangle_data, triangle_meta_data, bvh_data] = intersectable_manager.build();
    triangle_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_data);
    tri_meta_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_meta_data);
    bvh_buf = accelerator.create_buffer(MemFlags::READ_ONLY, bvh_data);

    MaterialData material_data = material_loader.build();
    // Create a dummy array
    if (material_data.num_materials == 0) {
      material_data.data.emplace_back();
    }
    material_ims = accelerator.create_image2D_array(
      ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::WRAP, FilterMode::LINEAR, true,
      std::max(material_data.num_materials, static_cast<size_t>(1)),
      std::max(material_data.width, 1U), std::max(material_data.height, 1U), material_data.data);

    loaded_model = model_path;
  }

  // Update Sky
  const std::string& sky_path = scene.get_sky_path();
  if (sky_path != loaded_sky) {
    sky_loader.load_sky(sky_path.c_str());

    image_utils::image<float4> sky_data = sky_loader.build();

    // Create a dummy image
    sky_im = accelerator.create_image2D_read(ImageChannelOrder::RGBA, ImageChannelType::FLOAT,
                                             AddressMode::WRAP, FilterMode::LINEAR, true,
                                             sky_data.width, sky_data.height, sky_data.data);

    loaded_sky = sky_path;
  }

  rem_pixels_buf = accelerator.create_buffer<uint32_t>(MemFlags::READ_WRITE, 0U);

  this->width = width;
  this->height = height;
}

image_utils::image<uchar4> Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  accelerator.write_buffer(rem_pixels_buf, 0U);
  {
    PROFILE_SECTION_START("Raytrace kernel");
    uint2 global_dims { width, height / 2 };
    uint2 local_dims { 8, 4 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_raytrace), global_dims, local_dims,
                            scene_params_wrapper, temp_pixel_im1.write_access(),
                            temp_pixel_im2.write_access(), pixel_dims_wrapper, triangle_buf,
                            tri_meta_buf, bvh_buf, material_ims, sky_im);
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Interpolate kernel");
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_interpolate), global_dims, local_dims,
                            temp_pixel_im1.read_access(), temp_pixel_im2.write_access(),
                            pixel_dims_wrapper, rem_pixels_buf, rem_coords_buf);
    PROFILE_SECTION_END();
  }
  {
    PROFILE_SECTION_START("Fill remaining kernel");
    uint32_t counter = accelerator.read_buffer(rem_pixels_buf);
    uint2 global_dims { counter, 1 };
    uint2 local_dims { 32, 1 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_fill_remaining), global_dims, local_dims,
                            scene_params_wrapper, temp_pixel_im2.write_access(), pixel_dims_wrapper,
                            triangle_buf, tri_meta_buf, bvh_buf, material_ims, sky_im,
                            rem_pixels_buf, rem_coords_buf);
    PROFILE_SECTION_END();
  }
  {
    PROFILE_SECTION_START("Postprocess kernel");
    uint2 global_dims { width, height };
    uint2 local_dims { 8, 4 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_post_process), global_dims, local_dims,
                            scene_params_wrapper, temp_pixel_im2.read_access(), pixel_im,
                            pixel_dims_wrapper);
    PROFILE_SECTION_END();
  }

  PROFILE_SECTION_START("Read image");
  std::vector<uchar4> pixels = accelerator.read_image2D(pixel_im, width, height);
  PROFILE_SECTION_END();

  return {
    std::move(pixels),
    width,
    height,
  };
}

}
