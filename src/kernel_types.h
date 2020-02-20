#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include <tuple>

#include "backend/types.h"

struct EyeCoords {
  float2 coord_scale;
  float2 coord_dims;
  float3 eye_pos;
  float3 eye_coord_frame0;
  float3 eye_coord_frame1;
  float3 eye_coord_frame2;
};

// Packed to 32 bytes to fit in a cache line
// The 4th element of top_offset_left contains either the triangle offset or the left index
// The 4th element of bottom_num_right contains either the number of triangles or the right index
// depending on whether or not the node is an inner node or a leaf node
// If the 4th element of bottom_num_right < 0, then the node is a leaf node
struct FlatBVHNode {
  float4 top_offset_left;
  float4 bottom_num_right;
};

// Woop 4x3 affine transform matrix
struct TriangleData {
  float4 transform_x;
  float4 transform_y;
  float4 transform_z;
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
  int32_t diffuse_index;
  int32_t metallic_index;
  int32_t roughness_index;
  int32_t ambient_occlusion_index;
  int32_t normal_index;
};

struct KernelConstants {
  uint32_t triangle_per_leaf_bits;
  float3 default_diffuse;
  float default_metallic;
  float default_roughness;
  float default_ambient_occlusion;
  float3 light_position;
  float3 light_intensity;
  uint32_t ray_recursion_depth;
};

using Dims = std::tuple<uint32_t, uint32_t, uint32_t>;

#endif // KERNEL_TYPES_H