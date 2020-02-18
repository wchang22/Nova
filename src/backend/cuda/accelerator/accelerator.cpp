#include "accelerator.h"
#include "constants.h"

Accelerator::Accelerator(const SceneParser& scene_parser)
{
  const auto [ default_diffuse, default_metallic, default_roughness, default_ambient_occlusion ]
    = scene_parser.get_shading_default_settings();
  const auto [ light_position, light_intensity ] = scene_parser.get_light_settings();
  const unsigned int ray_recursion_depth = scene_parser.get_ray_recursion_depth();

  kernel_constants = {
    TRIANGLES_PER_LEAF_BITS,
    { default_diffuse.x, default_diffuse.y, default_diffuse.z },
    default_metallic,
    default_roughness,
    default_ambient_occlusion,
    { light_position.x, light_position.y, light_position.z },
    { light_intensity.x, light_intensity.y, light_intensity.z },
    ray_recursion_depth,
  };
}

Image2D Accelerator::create_image2D(MemFlags mem_flags, ImageChannelOrder channel_order,
                                    ImageChannelType channel_type, size_t width, size_t height) 
                                    const {
  if (width == 0 || height == 0) {
    throw AcceleratorException("Cannot build an empty Image2D");
  }
}

Image2DArray Accelerator::create_image2D_array(MemFlags mem_flags, ImageChannelOrder channel_order,
                                               ImageChannelType channel_type, size_t array_size, 
                                               size_t width, size_t height) const {
  if (array_size == 0 || width == 0 || height == 0) {
    throw AcceleratorException("Cannot build an empty Image2DArray");
  }
}
