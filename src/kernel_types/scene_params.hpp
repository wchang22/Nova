#ifndef KERNEL_TYPE_SCENE_PARAMS_HPP
#define KERNEL_TYPE_SCENE_PARAMS_HPP

#include "backend/types.hpp"
#include "kernel_types/area_light.hpp"
#include "kernel_types/eye_coords.hpp"

namespace nova {

struct SceneParams {
  EyeCoords eye_coords;
  AreaLight light;
  float3 shading_diffuse;
  float shading_metallic;
  float shading_roughness;
  float shading_ambient_occlusion;
  int num_samples;
  float exposure;
  char anti_aliasing;
};

}

#endif // KERNEL_TYPE_SCENE_PARAMS_HPP