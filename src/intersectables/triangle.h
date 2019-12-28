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

#endif // TRIANGLE_H
