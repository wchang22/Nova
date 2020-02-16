#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <glm/glm.hpp>
#include <fstream>

#include "backend/types.h"
#include "intersectables/aabb.h"

using namespace glm;

struct Triangle {
  vec3 v1;
  vec3 v2;
  vec3 v3;

  AABB get_bounds() const;
  bool operator==(const Triangle& t) const;
};

struct TriangleHash {
  size_t operator()(const Triangle& tri) const;
};

std::istream& operator>>(std::istream& in, Triangle& tri);
std::ostream& operator<<(std::ostream& out, const Triangle& tri);

// Woop 4x3 affine transform matrix
struct TriangleData {
  float4 transform_x;
  float4 transform_y;
  float4 transform_z;
};

struct TriangleMeta {
  vec3 normal1;
  vec3 normal2;
  vec3 normal3;
  vec3 tangent1;
  vec3 tangent2;
  vec3 tangent3;
  vec3 bitangent1;
  vec3 bitangent2;
  vec3 bitangent3;
  vec2 texture_coord1;
  vec2 texture_coord2;
  vec2 texture_coord3;
  int diffuse_index;
  int metallic_index;
  int roughness_index;
  int ambient_occlusion_index;
  int normal_index;
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

#endif // TRIANGLE_H
