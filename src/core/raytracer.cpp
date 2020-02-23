#include "raytracer.h"
#include "util/image/imageutils.h"
#include "util/profiling/profiling.h"
#include "model/model.h"
#include "constants.h"

REGISTER_KERNEL(kernel_raytrace)

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

  ADD_KERNEL(accelerator, kernel_raytrace)
}

void Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  PROFILE_SECTION_START("Build data");
  auto image = accelerator.create_image2D_write<uchar4>(
    ImageChannelOrder::RGBA, ImageChannelType::UINT8, width, height);
  std::vector<uchar4> image_buf;

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
  
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Raytrace profile");
  for (int i = 0; i < NUM_PROFILE_ITERATIONS; i++) {
    PROFILE_SCOPE("Raytrace profile loop");

    PROFILE_SECTION_START("Enqueue kernel");
    uint3 global_dims = { width, height, 1 };
    CALL_KERNEL(accelerator, kernel_raytrace, global_dims,
                image, ec, triangle_buf, tri_meta_buf, bvh_buf, material_ims)
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Read image");
    image_buf = accelerator.read_image(image, width, height);
    PROFILE_SECTION_END();
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write image");
  image_utils::write_image((name + ".jpg").c_str(), { image_buf, width, height });
  PROFILE_SECTION_END();
}
