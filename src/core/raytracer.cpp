#include "raytracer.hpp"
#include "constants.hpp"
#include "model/model.hpp"
#include "util/image/imageutils.hpp"
#include "util/profiling/profiling.hpp"

Raytracer::Raytracer(uint32_t width, uint32_t height, const std::string& name)
  : width(width),
    height(height),
    name(name),
    camera_settings(scene_parser.get_camera_settings()),
    camera(camera_settings.position,
           camera_settings.forward,
           camera_settings.up,
           width,
           height,
           camera_settings.fovy),
    intersectable_manager(name),
    accelerator(scene_parser) {
  const auto model_paths = scene_parser.get_model_paths();
  for (const auto& model_path : model_paths) {
    Model model(model_path, material_loader);
    intersectable_manager.add_model(model);
  }

  accelerator.add_kernel("kernel_raytrace");
  accelerator.add_kernel("kernel_interpolate");
  accelerator.add_kernel("kernel_fill_remaining");
}

void Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  PROFILE_SECTION_START("Build data");
  auto pixel_buf = accelerator.create_buffer<uchar4>(MemFlags::READ_WRITE, width * height);
  accelerator.fill_buffer(pixel_buf, width * height, {});
  auto pixel_dims_wrapper = accelerator.create_wrapper<uint2>(uint2 { width, height });
  auto ec = accelerator.create_wrapper<EyeCoords>(camera.get_eye_coords());
  auto [triangle_data, triangle_meta_data, bvh_data] = intersectable_manager.build();
  auto triangle_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_data);
  auto tri_meta_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_meta_data);
  auto bvh_buf = accelerator.create_buffer(MemFlags::READ_ONLY, bvh_data);
  auto rem_coords_buf = accelerator.create_buffer<uint2>(MemFlags::READ_WRITE, width * height / 2);
  auto rem_pixels_buf = accelerator.create_buffer<uint32_t>(MemFlags::READ_WRITE, 0U);

  MaterialData material_data = material_loader.build();
  // Create a dummy array
  if (material_data.num_materials == 0) {
    material_data.data.emplace_back();
  }
  auto material_ims = accelerator.create_image2D_array(
    ImageChannelOrder::RGBA, ImageChannelType::UINT8, AddressMode::WRAP, FilterMode::NEAREST, true,
    std::max(material_data.num_materials, static_cast<size_t>(1)),
    std::max(material_data.width, 1U), std::max(material_data.height, 1U), material_data.data);
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Raytrace profile");
  for (int i = 0; i < NUM_PROFILE_ITERATIONS; i++) {
    PROFILE_SCOPE("Raytrace profile loop");

    accelerator.write_buffer(rem_pixels_buf, 0U);
    {
      PROFILE_SECTION_START("Raytrace kernel");
      uint2 global_dims { width, height / 2 };
      uint2 local_dims { 8, 4 };
      accelerator.call_kernel(RESOLVE_KERNEL(kernel_raytrace), global_dims, local_dims, pixel_buf,
                              pixel_dims_wrapper, ec, triangle_buf, tri_meta_buf, bvh_buf,
                              material_ims);
      PROFILE_SECTION_END();

      PROFILE_SECTION_START("Interpolate kernel");
      accelerator.call_kernel(RESOLVE_KERNEL(kernel_interpolate), global_dims, local_dims,
                              pixel_buf, pixel_dims_wrapper, ec, triangle_buf, tri_meta_buf,
                              bvh_buf, material_ims, rem_pixels_buf, rem_coords_buf);
      PROFILE_SECTION_END();
    }
    {
      PROFILE_SECTION_START("Fill remaining kernel");
      uint32_t counter = accelerator.read_buffer(rem_pixels_buf);
      uint2 global_dims { counter, 1 };
      uint2 local_dims { 32, 1 };
      accelerator.call_kernel(RESOLVE_KERNEL(kernel_fill_remaining), global_dims, local_dims,
                              pixel_buf, pixel_dims_wrapper, ec, triangle_buf, tri_meta_buf,
                              bvh_buf, material_ims, rem_pixels_buf, rem_coords_buf);
      PROFILE_SECTION_END();
    }
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Read image");
  std::vector<uchar4> pixels = accelerator.read_buffer(pixel_buf, width * height);
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write image");
  image_utils::write_image((name + ".jpg").c_str(), { pixels, width, height });
  PROFILE_SECTION_END();
}
