#ifndef KERNEL_TYPE_TRIANGLE_HPP
#define KERNEL_TYPE_TRIANGLE_HPP

#include "backend/types.hpp"
#include "kernel_types/matrix.hpp"

namespace nova {

// Woop 4x3 affine transform matrix
struct TriangleData {
  Mat3x4 transform;
};

struct TriangleMetaData {
  float3 normal1;
  float3 normal2;
  float3 normal3;
  float3 tangent1;
  float3 tangent2;
  float3 tangent3;
  float3 bitangent1;
  float3 bitangent2;
  float3 bitangent3;
  float2 texture_coord1;
  float2 texture_coord2;
  float2 texture_coord3;
  float3 kD;
  float3 kE;
  float metallic;
  float roughness;
  int diffuse_index;
  int metallic_index;
  int roughness_index;
  int normal_index;
  int light_index;
};

}

#endif // KERNEL_TYPE_TRIANGLE_HPP
