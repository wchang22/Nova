#include <stb_image.h>

#include "raytracer.h"
#include "util/image/imageutils.h"
#include "util/profiling/profiling.h"
#include "model/model.h"
#include "constants.h"

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

  ADD_KERNEL(accelerator, raytrace)
}

void Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  PROFILE_SECTION_START("Build data");
  Image2D image = accelerator.create_image2D(MemFlags::WRITE_ONLY, ImageChannelOrder::RGBA,
                                             ImageChannelType::UINT8, width, height);
  std::vector<uint8_t> image_buf;

  Wrapper<EyeCoords> ec = accelerator.create_wrapper<EyeCoords>(camera.get_eye_coords());

  auto [ triangle_data, triangle_meta_data, bvh_data ] = intersectable_manager.build();
  Buffer<TriangleData> triangle_buf = accelerator.create_buffer(MemFlags::READ_ONLY, triangle_data);
  Buffer<TriangleMetaData> tri_meta_buf =
    accelerator.create_buffer(MemFlags::READ_ONLY, triangle_meta_data);
  Buffer<FlatBVHNode> bvh_buf = accelerator.create_buffer(MemFlags::READ_ONLY, bvh_data);

  MaterialData material_data = material_loader.build();
  Image2DArray material_ims;
  // Create a dummy array if size 0
  if (material_data.num_materials == 0) {
    material_ims = accelerator.create_image2D_array(
      MemFlags::READ_ONLY,
      ImageChannelOrder::RGBA, ImageChannelType::UINT8,
      1, 1, 1
    );
  } else {
    material_ims = accelerator.create_image2D_array(
      MemFlags::READ_ONLY,
      ImageChannelOrder::RGBA, ImageChannelType::UINT8,
      material_data.num_materials, material_data.width, material_data.height, material_data.data
    );
  }
  
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Raytrace profile");
  for (int i = 0; i < NUM_PROFILE_ITERATIONS; i++) {
    PROFILE_SCOPE("Raytrace profile loop");

    PROFILE_SECTION_START("Enqueue kernel");
    CALL_KERNEL(accelerator, raytrace, std::make_tuple(width, height, 1), {},
                image, ec, triangle_buf, tri_meta_buf, bvh_buf, material_ims)
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Read image");
    image_buf = accelerator.read_image<uint8_t>(image, width, height, STBI_rgb_alpha);
    PROFILE_SECTION_END();
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write image");
  image_utils::write_image((name + ".jpg").c_str(), { image_buf, width, height });
  PROFILE_SECTION_END();
}
