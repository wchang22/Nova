#include <glm/gtx/vec_swizzle.hpp>

#include "aabb.h"

float AABB::get_surface_area() const {
  vec3 dims = top - bottom;
  return dot(xyz(dims), yzx(dims)) * 2;
}

float AABB::get_cost(size_t num_triangles) const {
  return get_surface_area() * num_triangles;
}

vec3 AABB::get_center() const {
  return (top + bottom) / 2.0f;
}

void AABB::grow(const AABB& other) {
  top = max(top, other.top);
  bottom = min(bottom, other.bottom);
}

void AABB::shrink(const AABB& other) {
  top = min(top, other.top);
  bottom = max(bottom, other.bottom);
}

bool AABB::intersects(const AABB& other, int axis) const {
  return top[axis] > other.bottom[axis] || other.top[axis] > bottom[axis];
}

AABB AABB::get_intersection(const AABB& other) const {
  return {
    min(top, other.top),
    max(bottom, other.bottom)
  };
}

bool AABB::operator==(const AABB& other) const {
  return top == other.top && bottom == other.bottom;
}
