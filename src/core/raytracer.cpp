#include "raytracer.hpp"
#include "camera/camera.hpp"
#include "constants.hpp"
#include "model/model.hpp"
#include "scene/scene.hpp"
#include "util/profiling/profiling.hpp"

Raytracer::Raytracer() {
  accelerator.add_kernel("kernel_raytrace");
  accelerator.add_kernel("kernel_interpolate");
  accelerator.add_kernel("kernel_fill_remaining");
}

void Raytracer::set_scene(const Scene& scene) {
  PROFILE_SCOPE("Set Scene");

  uint32_t width = static_cast<uint32_t>(scene.get_dimensions()[0]);
  uint32_t height = static_cast<uint32_t>(scene.get_dimensions()[1]);

  // Update Scene Params
  const auto& shading_diffuse = scene.get_shading_diffuse();
  const auto& shading_metallic = scene.get_shading_metallic();
  const auto& shading_roughness = scene.get_shading_roughness();
  const auto& shading_ambient_occlusion = scene.get_shading_ambient_occlusion();
  const auto& light_position = scene.get_light_position();
  const auto& light_intensity = scene.get_light_intensity();
  const auto ray_bounces = scene.get_ray_bounces();

  scene_params_wrapper = accelerator.create_wrapper<SceneParams>(
    SceneParams { scene.get_camera_eye_coords(),
                  { light_position[0], light_position[1], light_position[2] },
                  { light_intensity[0], light_intensity[1], light_intensity[2] },
                  { shading_diffuse[0], shading_diffuse[1], shading_diffuse[2] },
                  shading_metallic,
                  shading_roughness,
                  shading_ambient_occlusion,
                  ray_bounces });

  // Update buffers depending on width, height
  if (this->width != width || this->height != height) {
    pixel_buf = accelerator.create_buffer<uchar4>(MemFlags::READ_WRITE, width * height);
    pixel_dims_wrapper = accelerator.create_wrapper<uint2>(uint2 { width, height });
    rem_coords_buf =
      accelerator.create_buffer<uint2>(MemFlags::READ_WRITE, std::max(width * height / 2, 1U));
  }

  // Update Model
  const auto model_path = scene.get_model_path();
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
      ImageChannelOrder::RGBA, ImageChannelType::UINT8, AddressMode::WRAP, FilterMode::NEAREST,
      true, std::max(material_data.num_materials, static_cast<size_t>(1)),
      std::max(material_data.width, 1U), std::max(material_data.height, 1U), material_data.data);

    loaded_model = model_path;
  }

  rem_pixels_buf = accelerator.create_buffer<uint32_t>(MemFlags::READ_WRITE, 0U);

  this->width = width;
  this->height = height;
}

image_utils::image Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  accelerator.write_buffer(rem_pixels_buf, 0U);
  {
    PROFILE_SECTION_START("Raytrace kernel");
    uint2 global_dims { width, height / 2 };
    uint2 local_dims { 8, 4 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_raytrace), global_dims, local_dims,
                            scene_params_wrapper, pixel_buf, pixel_dims_wrapper, triangle_buf,
                            tri_meta_buf, bvh_buf, material_ims);
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Interpolate kernel");
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_interpolate), global_dims, local_dims, pixel_buf,
                            pixel_dims_wrapper, triangle_buf, tri_meta_buf, bvh_buf, material_ims,
                            rem_pixels_buf, rem_coords_buf);
    PROFILE_SECTION_END();
  }
  {
    PROFILE_SECTION_START("Fill remaining kernel");
    uint32_t counter = accelerator.read_buffer(rem_pixels_buf);
    uint2 global_dims { counter, 1 };
    uint2 local_dims { 32, 1 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_fill_remaining), global_dims, local_dims,
                            scene_params_wrapper, pixel_buf, pixel_dims_wrapper, triangle_buf,
                            tri_meta_buf, bvh_buf, material_ims, rem_pixels_buf, rem_coords_buf);
    PROFILE_SECTION_END();
  }

  PROFILE_SECTION_START("Read image");
  std::vector<uchar4> pixels = accelerator.read_buffer(pixel_buf, width * height);
  PROFILE_SECTION_END();

  return {
    pixels,
    width,
    height,
  };
}
