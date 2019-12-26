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

struct TriangleData {
  cl_float3 vertex;
  cl_float3 edge1;
  cl_float3 edge2;
};

#endif // TRIANGLE_H