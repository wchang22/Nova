#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <CL/cl2.hpp>
#include <glm/glm.hpp>

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
  cl_float4 transform_x;
  cl_float4 transform_y;
  cl_float4 transform_z;
};

struct TriangleMeta {
  vec3 normal1;
  vec3 normal2;
  vec3 normal3;
  vec2 texture_coord1;
  vec2 texture_coord2;
  vec2 texture_coord3;
  bool has_textures;
  int ambient_index;
  int diffuse_index;
  int specular_index;
};

struct TriangleMetaData {
  cl_float3 normal1;
  cl_float3 normal2;
  cl_float3 normal3;
  cl_float2 texture_coord1;
  cl_float2 texture_coord2;
  cl_float2 texture_coord3;
  cl_bool has_textures;
  cl_int ambient_index;
  cl_int diffuse_index;
  cl_int specular_index;
};

#endif // TRIANGLE_H
