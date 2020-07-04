#ifndef KERNEL_TYPE_SCENE_PARAMS_HPP
#define KERNEL_TYPE_SCENE_PARAMS_HPP

#include "backend/types.hpp"
#include "kernel_types/eye_coords.hpp"

namespace nova {

struct SceneParams {
  EyeCoords eye_coords;
  float3 light_position;
  float3 light_intensity;
  float3 shading_diffuse;
  float shading_metallic;
  float shading_roughness;
  float shading_ambient_occlusion;
  int ray_bounces;
  float exposure;
  char anti_aliasing;
};

}

#endif // KERNEL_TYPE_SCENE_PARAMS_HPP