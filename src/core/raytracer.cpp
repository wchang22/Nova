#include <chrono>
#include <iostream>

#include "constants.hpp"
#include "kernel_types/area_light.hpp"
#include "model/model.hpp"
#include "raytracer.hpp"
#include "scene/scene.hpp"
#include "util/exception/exception.hpp"
#include "util/profiling/profiling.hpp"
#include "vector/vector_conversions.hpp"
#include "vector/vector_utils.hpp"

namespace nova {

Raytracer::Raytracer() {
  accelerator.add_kernel("kernel_generate");
  accelerator.add_kernel("kernel_intersect");
  accelerator.add_kernel("kernel_extend");
  accelerator.add_kernel("kernel_write");

  accelerator.add_kernel("kernel_raytrace");
  accelerator.add_kernel("kernel_accumulate");
  accelerator.add_kernel("kernel_post_process");

  denoise_device = oidn::newDevice();
  denoise_device.commit();

  const char* errorMessage;
  oidn::Error denoise_error = denoise_device.getError(errorMessage);
  switch (denoise_error) {
    case oidn::Error::UnsupportedHardware:
      std::cerr << "No device found capable of using Intel Open Image Denoise" << std::endl;
      denoise_available = false;
      break;
    case oidn::Error::None:
      denoise_available = true;
      break;
    default:
      throw DenoiseException(errorMessage);
  }

  denoise_filter = denoise_device.newFilter("RT");
  denoise_filter.set("hdr", true);
}

void Raytracer::start() {
  sample_index = 0;
  accelerator.fill_image2D(prev_color_img, width, height, float4 {});
}

void Raytracer::set_scene(const Scene& scene) {
  PROFILE_SCOPE("Set Scene");

  const uint32_t width = static_cast<uint32_t>(scene.get_output_dimensions()[0]);
  const uint32_t height = static_cast<uint32_t>(scene.get_output_dimensions()[1]);
  size_t ray_count = width * height;

  // Update Scene Params
  const EyeCoords& eye_coords = scene.get_camera_eye_coords();
  const float3 shading_diffuse = vec_to_float3(scene.get_shading_diffuse());
  const float shading_metallic = scene.get_shading_metallic();
  const float shading_roughness = scene.get_shading_roughness();
  const bool path_tracing = scene.get_path_tracing();
  const int num_samples = scene.get_num_samples();
  const float exposure = scene.get_exposure();
  const bool anti_aliasing = scene.get_anti_aliasing();

  scene_params_wrapper = accelerator.create_wrapper<SceneParams>(
    SceneParams { eye_coords, shading_diffuse, shading_metallic, shading_roughness, path_tracing,
                  num_samples, exposure, anti_aliasing });

  // Update buffers depending on width, height
  if (this->width != width || this->height != height) {
    color_img = accelerator.create_image2D_write<uchar4>(ImageChannelOrder::RGBA,
                                                         ImageChannelType::UINT8, width, height);
    prev_color_img = accelerator.create_image2D_read<float4>(
      ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
      true, width, height);
    temp_color_img1 = accelerator.create_image2D_readwrite<float4>(
      ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
      true, width, height);
    temp_color_img2 = accelerator.create_image2D_readwrite<float4>(
      ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
      true, width, height);

    if (denoise_available) {
      albedo_img1 = accelerator.create_image2D_readwrite<float4>(
        ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
        true, width, height);
      normal_img1 = accelerator.create_image2D_readwrite<float4>(
        ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
        true, width, height);
      albedo_img2 = accelerator.create_image2D_readwrite<float4>(
        ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
        true, width, height);
      normal_img2 = accelerator.create_image2D_readwrite<float4>(
        ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
        true, width, height);
      prev_albedo_img = accelerator.create_image2D_read<float4>(
        ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
        true, width, height);
      prev_normal_img = accelerator.create_image2D_read<float4>(
        ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::CLAMP, FilterMode::LINEAR,
        true, width, height);
    }

    pixel_dims_wrapper = accelerator.create_wrapper<uint2>(uint2 { width, height });
  }

  // Update Model, lights, ground
  // TODO: Make this so don't need to regenerate everything
  const std::string& model_path = scene.get_model_path();
  const std::vector<AreaLight>& lights = scene.get_lights();
  const std::optional<GroundPlane>& ground_plane = scene.get_ground_plane();

  if (model_path != loaded_model || lights != loaded_lights ||
      ground_plane != loaded_ground_plane) {
    intersectable_manager.clear();
    material_loader.clear();

    Model model(model_path, material_loader);
    intersectable_manager.add_model(model);
    for (const auto& light : lights) {
      intersectable_manager.add_light(light);
    }
    if (ground_plane.has_value()) {
      intersectable_manager.add_ground_plane(ground_plane.value());
    }

    auto [triangle_data, triangle_meta_data, bvh_data, light_data] = intersectable_manager.build();
    triangle_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_data);
    tri_meta_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_meta_data);
    bvh_buf = accelerator.create_buffer(MemFlags::READ_ONLY, bvh_data);
    if (!light_data.empty()) {
      light_buf = accelerator.create_buffer(MemFlags::READ_ONLY, light_data);
    }
    num_lights_wrapper = accelerator.create_wrapper<uint32_t>(light_data.size());

    MaterialData material_data = material_loader.build();
    // Create a dummy array
    if (material_data.num_materials == 0) {
      material_data.data.emplace_back();
    }
    material_imgs = accelerator.create_image2D_array(
      ImageChannelOrder::RGBA, ImageChannelType::FLOAT, AddressMode::WRAP, FilterMode::LINEAR, true,
      std::max(material_data.num_materials, static_cast<size_t>(1)),
      std::max(material_data.width, 1U), std::max(material_data.height, 1U), material_data.data);

    loaded_model = model_path;
    loaded_lights = lights;
    loaded_ground_plane = ground_plane;
  }

  // Update Sky
  const std::string& sky_path = scene.get_sky_path();
  if (sky_path != loaded_sky) {
    sky_loader.load_sky(sky_path.c_str());

    image_utils::image<float4> sky_data = sky_loader.build();

    // Create a dummy image
    sky_img = accelerator.create_image2D_read(ImageChannelOrder::RGBA, ImageChannelType::FLOAT,
                                              AddressMode::WRAP, FilterMode::LINEAR, true,
                                              sky_data.width, sky_data.height, sky_data.data);

    loaded_sky = sky_path;
  }

  ray_buf = accelerator.create_buffer<PackedRay>(MemFlags::READ_WRITE, ray_count);
  extended_ray_buf = accelerator.create_buffer<PackedRay>(MemFlags::READ_WRITE, ray_count);
  path_buf = accelerator.create_buffer<Path>(MemFlags::READ_WRITE, ray_count);
  intersection_buf = accelerator.create_buffer<IntersectionData>(MemFlags::READ_WRITE, ray_count);
  ray_count_buf = accelerator.create_buffer<uint32_t>(MemFlags::READ_WRITE, 0U);
  intersection_count_buf = accelerator.create_buffer<uint32_t>(MemFlags::READ_WRITE, 0U);

  this->width = width;
  this->height = height;
  this->num_samples = num_samples;
}

image_utils::image<uchar4> Raytracer::raytrace(bool denoise) {
  PROFILE_SCOPE("Raytrace");

  // Create sample specific data
  auto sample_index_wrapper = accelerator.create_wrapper<int>(sample_index);
  using namespace std::chrono;
  auto time_wrapper = accelerator.create_wrapper<uint32_t>(
    duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
  auto denoise_available_wrapper = accelerator.create_wrapper<uint32_t>(denoise_available);

  uint32_t ray_count = width * height;
  uint32_t intersection_count = 0;

  {
    PROFILE_SECTION_START("Generate kernel");
    uint2 global_dims { width, height };
    uint2 local_dims { 8, 4 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_generate), global_dims, local_dims, ray_buf,
                            path_buf, scene_params_wrapper, sample_index_wrapper, time_wrapper,
                            pixel_dims_wrapper);
    PROFILE_SECTION_END();
  }

  PROFILE_SECTION_START("Wavefront");
  while (ray_count > 0) {
    PROFILE_SCOPE("Wavefront loop");
    printf("%d ", ray_count);
    auto t0 = steady_clock::now();

    {
      PROFILE_SECTION_START("Intersect kernel");
      uint2 global_dims { ray_count, 1 };
      uint2 local_dims { 32, 1 };

      auto ray_count_wrapper = accelerator.create_wrapper<uint32_t>(ray_count);
      accelerator.write_buffer(intersection_count_buf, 0U);

      accelerator.call_kernel(RESOLVE_KERNEL(kernel_intersect), global_dims, local_dims, ray_buf,
                              path_buf, intersection_buf, intersection_count_buf, ray_count_wrapper,
                              denoise_available_wrapper, triangle_buf, bvh_buf, sky_img);
      PROFILE_SECTION_END();
    }
    printf("%ld ", duration_cast<milliseconds>(steady_clock::now() - t0).count());

    auto t1 = steady_clock::now();
    {
      PROFILE_SECTION_START("Extend kernel");
      intersection_count = accelerator.read_buffer(intersection_count_buf);
      uint2 global_dims { intersection_count, 1 };
      uint2 local_dims { 32, 1 };

      auto intersection_count_wrapper = accelerator.create_wrapper<uint32_t>(intersection_count);
      accelerator.write_buffer(ray_count_buf, 0U);

      accelerator.call_kernel(RESOLVE_KERNEL(kernel_extend), global_dims, local_dims, ray_buf,
                              intersection_buf, path_buf, extended_ray_buf, ray_count_buf,
                              intersection_count_wrapper, scene_params_wrapper, time_wrapper,
                              denoise_available_wrapper, triangle_buf, tri_meta_buf, bvh_buf,
                              light_buf, num_lights_wrapper, material_imgs);
      PROFILE_SECTION_END();
    }
    printf("%ld ", duration_cast<milliseconds>(steady_clock::now() - t1).count());

    ray_count = accelerator.read_buffer(ray_count_buf);
    std::swap(ray_buf, extended_ray_buf);

    printf("%ld\n", duration_cast<milliseconds>(steady_clock::now() - t0).count());
  }
  PROFILE_SECTION_END();

  {
    PROFILE_SECTION_START("Write kernel");
    uint2 global_dims { width, height };
    uint2 local_dims { 8, 4 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_write), global_dims, local_dims, path_buf,
                            temp_color_img1.write_access(), albedo_img1.write_access(),
                            normal_img1.write_access(), sample_index_wrapper,
                            denoise_available_wrapper, pixel_dims_wrapper);
    PROFILE_SECTION_END();
  }

  // {
  //   PROFILE_SECTION_START("Raytrace kernel");
  //   uint2 global_dims { width, height };
  //   uint2 local_dims { 8, 4 };
  //   accelerator.call_kernel(RESOLVE_KERNEL(kernel_raytrace), global_dims, local_dims,
  //                           scene_params_wrapper, time_wrapper, temp_color_img1.write_access(),
  //                           pixel_dims_wrapper, triangle_buf, tri_meta_buf, bvh_buf, light_buf,
  //                           num_lights_wrapper, material_imgs, sky_img,
  //                           denoise_available_wrapper, albedo_img1.write_access(),
  //                           normal_img1.write_access());
  //   PROFILE_SECTION_END();
  // }

  {
    PROFILE_SECTION_START("Accumulate kernel");
    uint2 global_dims { width, height };
    uint2 local_dims { 8, 4 };
    accelerator.call_kernel(
      RESOLVE_KERNEL(kernel_accumulate), global_dims, local_dims, sample_index_wrapper,
      denoise_available_wrapper, temp_color_img1.read_access(), albedo_img1.read_access(),
      normal_img1.read_access(), prev_color_img, prev_albedo_img, prev_normal_img,
      temp_color_img2.write_access(), albedo_img2.write_access(), normal_img2.write_access(),
      pixel_dims_wrapper);
    PROFILE_SECTION_END();
  }
  {
    PROFILE_SECTION_START("Copy previous image");
    accelerator.copy_image2D(prev_color_img, temp_color_img2, width, height);
    if (denoise_available) {
      accelerator.copy_image2D(prev_albedo_img, albedo_img2, width, height);
      accelerator.copy_image2D(prev_normal_img, normal_img2, width, height);
    }
    PROFILE_SECTION_END();
  }
  if (denoise_available && denoise && scene_params_wrapper.data().path_tracing) {
    PROFILE_SECTION_START("Denoise: read");
    std::vector<float> output_buffer(width * height * 3);

    std::vector<float4> color_image = accelerator.read_image2D(temp_color_img2, width, height);
    std::vector<float4> albedo_feature_image = accelerator.read_image2D(albedo_img2, width, height);
    std::vector<float4> normal_feature_image = accelerator.read_image2D(normal_img2, width, height);

    std::vector<float> color_buffer = flatten<float4, float, 3>(color_image);
    std::vector<float> albedo_buffer = flatten<float4, float, 3>(albedo_feature_image);
    std::vector<float> normal_buffer = flatten<float4, float, 3>(normal_feature_image);
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Denoise: filter");
    denoise_filter.setImage("color", color_buffer.data(), oidn::Format::Float3, width, height);
    denoise_filter.setImage("albedo", albedo_buffer.data(), oidn::Format::Float3, width, height);
    denoise_filter.setImage("normal", normal_buffer.data(), oidn::Format::Float3, width, height);
    denoise_filter.setImage("output", output_buffer.data(), oidn::Format::Float3, width, height);
    denoise_filter.commit();
    denoise_filter.execute();

    const char* errorMessage;
    if (denoise_device.getError(errorMessage) != oidn::Error::None) {
      throw DenoiseException(errorMessage);
    }
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Denoise: write");
    color_image = pack<float4, float, 3>(output_buffer);
    accelerator.write_image2D(temp_color_img2, width, height, color_image);
    PROFILE_SECTION_END();
  }
  {
    PROFILE_SECTION_START("Postprocess kernel");
    uint2 global_dims { width, height };
    uint2 local_dims { 8, 4 };
    accelerator.call_kernel(RESOLVE_KERNEL(kernel_post_process), global_dims, local_dims,
                            scene_params_wrapper, temp_color_img2.read_access(), color_img,
                            pixel_dims_wrapper);
    PROFILE_SECTION_END();
  }

  PROFILE_SECTION_START("Read image");
  std::vector<uchar4> pixels = accelerator.read_image2D(color_img, width, height);
  PROFILE_SECTION_END();

  return {
    pixels,
    width,
    height,
  };
}

}
