#ifndef AREA_LIGHT_HPP
#define AREA_LIGHT_HPP

#include <glm/glm.hpp>

namespace nova {

struct AreaLight {
  glm::vec3 intensity;
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 dims;

  inline bool operator==(const AreaLight& a) const {
    return intensity == a.intensity && position == a.position && normal == a.normal &&
           dims == a.dims;
  }
  inline bool operator!=(const AreaLight& a) const { return !(*this == a); }
};

}

#endif // AREA_LIGHT_HPP
