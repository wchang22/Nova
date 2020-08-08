#ifndef GROUND_PLANE_HPP
#define GROUND_PLANE_HPP

#include <glm/glm.hpp>

namespace nova {

struct GroundPlane {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 dims;
  glm::vec3 diffuse;
  float metallic;
  float roughness;

  inline bool operator==(const GroundPlane& other) const {
    return position == other.position && normal == other.normal && dims == other.dims &&
           diffuse == other.diffuse && metallic == other.metallic && roughness == other.roughness;
  }
  inline bool operator!=(const GroundPlane& other) const { return !(*this == other); }
};

}

#endif // GROUND_PLANE_HPP
