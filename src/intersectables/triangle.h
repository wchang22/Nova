#ifndef TRIANGLE_H
#define TRIANGLE_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

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
  vec3 tangent1;
  vec3 tangent2;
  vec3 tangent3;
  vec3 bitangent1;
  vec3 bitangent2;
  vec3 bitangent3;
  vec2 texture_coord1;
  vec2 texture_coord2;
  vec2 texture_coord3;
  int ambient_index;
  int diffuse_index;
  int specular_index;
  int normal_index;
};

struct TriangleMetaData {
  cl_float3 normal1;
  cl_float3 normal2;
  cl_float3 normal3;
  cl_float3 tangent1;
  cl_float3 tangent2;
  cl_float3 tangent3;
  cl_float3 bitangent1;
  cl_float3 bitangent2;
  cl_float3 bitangent3;
  cl_float2 texture_coord1;
  cl_float2 texture_coord2;
  cl_float2 texture_coord3;
  cl_int ambient_index;
  cl_int diffuse_index;
  cl_int specular_index;
  cl_int normal_index;
};

#endif // TRIANGLE_H
