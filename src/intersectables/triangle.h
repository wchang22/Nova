#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <CL/cl2.hpp>
#include <glm/glm.hpp>

using namespace glm;

struct Triangle {
  vec3 v1;
  vec3 v2;
  vec3 v3;

  std::pair<vec3, vec3> get_bounds() const;
  bool operator==(const Triangle& t) const;
};

struct TriangleHash {
  size_t operator()(const Triangle& tri) const;
};

struct TriangleData {
  cl_float3 vertex;
  cl_float3 normal;
  cl_float3 edge1;
  cl_float3 edge2;
};

#endif // TRIANGLE_H