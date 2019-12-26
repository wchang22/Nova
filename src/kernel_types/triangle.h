#ifndef KERNEL_TYPE_TRIANGLE_H
#define KERNEL_TYPE_TRIANGLE_H

#include "backend/types.h"
#include "kernel_types/matrix.h"

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
  int diffuse_index;
  int metallic_index;
  int roughness_index;
  int ambient_occlusion_index;
  int normal_index;
};

#endif // KERNEL_TYPE_TRIANGLE_H