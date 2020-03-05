#include "raytracer.h"
#include "util/image/imageutils.h"
#include "util/profiling/profiling.h"
#include "model/model.h"
#include "kernel_types/ray.h"
#include "kernel_types/intersection.h"
#include "constants.h"

#include <iostream>

REGISTER_KERNEL(kernel_raytrace)
REGISTER_KERNEL(kernel_intersect_rays)
REGISTER_KERNEL(kernel_shade_pixels)

Raytracer::Raytracer(uint32_t width, uint32_t height, const std::string& name)
  : width(width), height(height),
    name(name),
    camera_settings(scene_parser.get_camera_settings()),
    camera(camera_settings.position, camera_settings.forward, camera_settings.up,
           width, height, camera_settings.fovy),
    intersectable_manager(name),
    accelerator(scene_parser)
{
  const auto model_paths = scene_parser.get_model_paths();
  for (const auto& model_path : model_paths) {
    Model model(model_path, material_loader);
    intersectable_manager.add_model(model);
  }

  ADD_KERNEL(accelerator, kernel_generate_rays)
  ADD_KERNEL(accelerator, kernel_intersect_rays)
  ADD_KERNEL(accelerator, kernel_shade_pixels)
}

void Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  PROFILE_SECTION_START("Build data");
  std::vector<float3> image_buf;
  auto ec = accelerator.create_wrapper<EyeCoords>(camera.get_eye_coords());

  auto [ triangle_data, triangle_meta_data, bvh_data ] = intersectable_manager.build();
  auto triangle_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_data);
  auto tri_meta_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_meta_data);
  auto bvh_buf = accelerator.create_buffer(MemFlags::READ_ONLY, bvh_data);

  MaterialData material_data = material_loader.build();
  // Create a dummy array
  if (material_data.num_materials == 0) {
    material_data.data.emplace_back();
  }
  auto material_ims = accelerator.create_image2D_array(
    ImageChannelOrder::RGBA, ImageChannelType::UINT8,
    AddressMode::WRAP, FilterMode::NEAREST, true,
    std::max(material_data.num_materials, static_cast<size_t>(1)),
    std::max(material_data.width, 1U),
    std::max(material_data.height, 1U),
    material_data.data
  );

  uint32_t global_size = width * height;
  auto ray_buf = accelerator.create_buffer<PackedRay>(MemFlags::READ_WRITE, global_size);
  auto reflection_ray_buf = accelerator.create_buffer<PackedRay>(MemFlags::READ_WRITE, global_size);
  auto intersection_buf = accelerator.create_buffer<Intersection>(MemFlags::READ_WRITE,
                                                                  global_size);
  auto color_buf = accelerator.create_buffer<float3>(MemFlags::READ_WRITE, global_size);
  auto reflectance_buf = accelerator.create_buffer<float3>(MemFlags::READ_WRITE, global_size);
  uint32_t global_count = 0;
  auto global_count_buf = accelerator.create_buffer(MemFlags::WRITE_ONLY, global_count);

  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Raytrace profile");
  for (int i = 0; i < NUM_PROFILE_ITERATIONS; i++) {
    PROFILE_SCOPE("Raytrace profile loop");
    global_size = width * height;

    PROFILE_SECTION_START("Generate rays");
    accelerator.fill_buffer<float3>(color_buf, global_size, float3 { 0.0f, 0.0f, 0.0f });
    accelerator.fill_buffer<float3>(reflectance_buf, global_size, float3 { 1.0f, 1.0f, 1.0f });
    CALL_KERNEL(accelerator, kernel_generate_rays, { width, height, 1 }, { 8, 4, 1 },
                ray_buf, ec);
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Raytrace bounce loop");
    for (uint32_t depth = 0; depth < 5; depth++) {
      PROFILE_SCOPE("Raytrace bounce iteration");

      PROFILE_SECTION_START("Intersect rays");
      CALL_KERNEL(accelerator, kernel_intersect_rays, { global_size, 1, 1 }, { 32, 1, 1 },
                  ray_buf, intersection_buf, global_count_buf, triangle_buf, bvh_buf);
      PROFILE_SECTION_END();

      PROFILE_SECTION_START("Read/write global count 1");
      global_size = accelerator.read_buffer(global_count_buf);
      accelerator.write_buffer(global_count_buf, 0U);
      PROFILE_SECTION_END();
      if (global_size == 0) {
        break;
      }

      PROFILE_SECTION_START("Shade pixels");
      CALL_KERNEL(accelerator, kernel_shade_pixels, { global_size, 1, 1 }, { 32, 1, 1 },
                  ray_buf, color_buf, reflectance_buf, intersection_buf,
                  reflection_ray_buf, global_count_buf,
                  triangle_buf, bvh_buf, tri_meta_buf, material_ims);
      PROFILE_SECTION_END();

      PROFILE_SECTION_START("Read/write global count 2");
      global_size = accelerator.read_buffer(global_count_buf);
      accelerator.write_buffer(global_count_buf, 0U);
      std::swap(reflection_ray_buf, ray_buf);
      PROFILE_SECTION_END();

      if (global_size == 0) {
        break;
      }
    }
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Read image");
    image_buf = accelerator.read_buffer(color_buf, width * height);
    PROFILE_SECTION_END();
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write image");
  std::vector<uchar4> image_char_buf;
  std::transform(image_buf.cbegin(), image_buf.cend(), std::back_inserter(image_char_buf),
  [](const auto& pixel) {
    return uchar4 {
      static_cast<unsigned char>(std::min(x(pixel), 1.0f) * 255.0f),
      static_cast<unsigned char>(std::min(y(pixel), 1.0f) * 255.0f),
      static_cast<unsigned char>(std::min(z(pixel), 1.0f) * 255.0f),
      255
    };
  });
  image_utils::write_image((name + ".jpg").c_str(), { image_char_buf, width, height });
  PROFILE_SECTION_END();
}
