#ifndef KERNEL_TYPE_KERNEL_CONSTANTS_HPP
#define KERNEL_TYPE_KERNEL_CONSTANTS_HPP

#include "backend/types.hpp"

struct KernelConstants {
  int triangle_per_leaf_bits;
  unsigned triangle_num_shift;
  unsigned triangle_offset_mask;
  float3 default_diffuse;
  float default_metallic;
  float default_roughness;
  float default_ambient_occlusion;
  float3 light_position;
  float3 light_intensity;
  int ray_recursion_depth;
};

#endif // KERNEL_TYPE_KERNEL_CONSTANTS_HPP