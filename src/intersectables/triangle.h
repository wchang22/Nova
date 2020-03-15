#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <glm/glm.hpp>
#include <glm/gtx/extended_min_max.hpp>
#include <fstream>

#include "intersectables/aabb.h"
#include "kernel_types/triangle.h"

using namespace glm;

struct Triangle {
  vec3 v1;
  vec3 v2;
  vec3 v3;

  inline AABB get_bounds() const {
    vec3 top = max(v1, v2, v3);
    vec3 bottom = min(v1, v2, v3);

    return { top, bottom };
  }

  inline bool operator==(const Triangle& t) const {
    return t.v1 == v1 && t.v2 == v2 && t.v3 == v3;
  }
};

struct TriangleHash {
  size_t operator()(const Triangle& tri) const;
};

std::istream& operator>>(std::istream& in, Triangle& tri);
std::ostream& operator<<(std::ostream& out, const Triangle& tri);

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

#endif // TRIANGLE_H
