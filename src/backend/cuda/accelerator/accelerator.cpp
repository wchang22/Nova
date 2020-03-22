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
    { default_diffuse.x, default_diffuse.y, default_diffuse.z },
    default_metallic,
    default_roughness,
    default_ambient_occlusion,
    { light_position.x, light_position.y, light_position.z },
    { light_intensity.x, light_intensity.y, light_intensity.z },
    ray_recursion_depth,
  };
}
