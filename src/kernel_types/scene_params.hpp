#ifndef KERNEL_TYPE_SCENE_PARAMS_HPP
#define KERNEL_TYPE_SCENE_PARAMS_HPP

#include "backend/types.hpp"
#include "kernel_types/eye_coords.hpp"

namespace nova {

struct SceneParams {
  EyeCoords eye_coords;
  float3 shading_diffuse;
  float shading_metallic;
  float shading_roughness;
  char path_tracing;
  int num_samples;
  float exposure;
  char anti_aliasing;
};

}

#endif // KERNEL_TYPE_SCENE_PARAMS_HPP