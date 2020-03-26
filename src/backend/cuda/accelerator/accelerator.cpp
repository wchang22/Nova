#include "accelerator.hpp"
#include "constants.hpp"

Accelerator::Accelerator(const SceneParser& scene_parser) {
  const auto [default_diffuse, default_metallic, default_roughness, default_ambient_occlusion] =
    scene_parser.get_shading_default_settings();
  const auto [light_position, light_intensity] = scene_parser.get_light_settings();
  const int ray_recursion_depth = scene_parser.get_ray_recursion_depth();

  kernel_constants = {
    TRIANGLES_PER_LEAF_BITS,
    TRIANGLE_NUM_SHIFT,
    TRIANGLE_OFFSET_MASK,
    { default_diffuse[0], default_diffuse[1], default_diffuse[2] },
    default_metallic,
    default_roughness,
    default_ambient_occlusion,
    { light_position[0], light_position[1], light_position[2] },
    { light_intensity[0], light_intensity[1], light_intensity[2] },
    ray_recursion_depth,
  };
}
