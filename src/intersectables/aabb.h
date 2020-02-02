#ifndef AABB_H
#define AABB_H

#include <glm/glm.hpp>
#include <ostream>

using namespace glm;

// Axis-aligned bounding box
struct AABB {
  vec3 top;
  vec3 bottom;

  float get_surface_area() const;
  float get_cost(size_t num_triangles) const;
  vec3 get_center() const;
  void grow(const AABB& other);
  void shrink(const AABB& other);
  bool intersects(const AABB& other, int axis) const;
  bool intersects(const AABB& other) const;
  bool is_in(const AABB& other) const;
  AABB get_intersection(const AABB& other) const;
  AABB get_union(const AABB& other) const;

  bool operator==(const AABB& other) const;
  bool operator!=(const AABB& other) const;
};

std::ostream& operator<<(std::ostream& out, const AABB& aabb);

#endif // AABB_H
