#ifndef KERNEL_TYPE_KERNEL_CONSTANTS_H
#define KERNEL_TYPE_KERNEL_CONSTANTS_H

#include "backend/types.h"

struct KernelConstants {
  int triangle_per_leaf_bits;
  float3 default_diffuse;
  float default_metallic;
  float default_roughness;
  float default_ambient_occlusion;
  float3 light_position;
  float3 light_intensity;
  int ray_recursion_depth;
};

#endif // KERNEL_TYPE_KERNEL_CONSTANTS_H