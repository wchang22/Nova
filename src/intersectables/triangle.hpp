#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/extended_min_max.hpp>

#include "intersectables/aabb.hpp"
#include "kernel_types/triangle.hpp"

namespace nova {

struct Triangle {
  glm::vec3 v1;
  glm::vec3 v2;
  glm::vec3 v3;

  inline AABB get_bounds() const {
    glm::vec3 top = glm::max(v1, v2, v3);
    glm::vec3 bottom = glm::min(v1, v2, v3);

    return { top, bottom };
  }

  inline bool operator==(const Triangle& t) const { return t.v1 == v1 && t.v2 == v2 && t.v3 == v3; }
};

struct TriangleHash {
  size_t operator()(const Triangle& tri) const;
};

std::istream& operator>>(std::istream& in, Triangle& tri);
std::ostream& operator<<(std::ostream& out, const Triangle& tri);

struct TriangleMeta {
  glm::vec3 normal1;
  glm::vec3 normal2;
  glm::vec3 normal3;
  glm::vec3 tangent1;
  glm::vec3 tangent2;
  glm::vec3 tangent3;
  glm::vec3 bitangent1;
  glm::vec3 bitangent2;
  glm::vec3 bitangent3;
  glm::vec2 texture_coord1;
  glm::vec2 texture_coord2;
  glm::vec2 texture_coord3;
  glm::vec3 kA;
  glm::vec3 kD;
  glm::vec3 kS;
  glm::vec3 kE;
  int diffuse_index;
  int metallic_index;
  int roughness_index;
  int ambient_occlusion_index;
  int normal_index;
};

}

#endif // TRIANGLE_HPP
