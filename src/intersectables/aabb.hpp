#ifndef AABB_HPP
#define AABB_HPP

#include <glm/glm.hpp>
#include <glm/gtx/vec_swizzle.hpp>

// Axis-aligned bounding box
struct AABB {
  glm::vec3 top;
  glm::vec3 bottom;

  inline float get_surface_area() const {
    glm::vec3 dims = top - bottom;
    return dot(xyz(dims), yzx(dims)) * 2;
  }

  inline float get_cost(size_t num_triangles) const {
    return get_surface_area() * num_triangles;
  }

  inline glm::vec3 get_center() const {
    return (top + bottom) / 2.0f;
  }

  inline void grow(const AABB& other) {
    top = max(top, other.top);
    bottom = min(bottom, other.bottom);
  }

  inline void shrink(const AABB& other) {
    top = min(top, other.top);
    bottom = max(bottom, other.bottom);
  }

  inline bool operator==(const AABB& other) const {
    return top == other.top && bottom == other.bottom;
  }

  inline static AABB make_no_intersection() {
    static glm::vec3 vec_max(std::numeric_limits<float>::max());
    return { -vec_max, vec_max };
  }
};

#endif // AABB_HPP
