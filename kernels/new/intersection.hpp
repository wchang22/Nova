#ifndef KERNEL_INTERSECTION_HPP
#define KERNEL_INTERSECTION_HPP

#include "kernel_types/triangle.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/new/matrix.hpp"
#include "kernels/new/types.hpp"

namespace nova {

// Use woop transformation to transform ray to unit triangle space
// http://www.sven-woop.de/papers/2004-GH-SaarCOR.pdf
DEVICE inline bool
intersects_triangle(const Ray& ray, Intersection& intrs, int tri_index, const TriangleData& tri) {
  // Transform ray to unit triangle space
  Ray woop_ray = ray;
  woop_ray.origin = tri.transform * ray.origin;
  woop_ray.direction = make_mat3x3(tri.transform) * ray.direction;

  float t = -woop_ray.origin.z / woop_ray.direction.z;
  if (t < 0.0f || t >= intrs.length) {
    return false;
  }

  float u = woop_ray.origin.x + t * woop_ray.direction.x;
  float v = woop_ray.origin.y + t * woop_ray.direction.y;
  float3 barycentric = { 1.0f - u - v, u, v };

  if (any(isless(barycentric, 0.0f))) {
    return false;
  }

  intrs.length = t;
  intrs.barycentric = barycentric;
  intrs.tri_index = tri_index;
  return true;
}

// AABB fast intersection for BVH
DEVICE inline bool intersects_aabb(const Ray& ray, const float3& top, const float3& bottom) {
  // Find slab bounds on AABB
  float3 t1 = top * ray.inv_direction + ray.nio;
  float3 t2 = bottom * ray.inv_direction + ray.nio;
  float3 tvmin = min(t1, t2);
  float3 tvmax = max(t1, t2);

  // Find tightest components of min and max
  float tmin = max(tvmin.x, max(tvmin.y, tvmin.z));
  float tmax = min(tvmax.x, min(tvmax.y, tvmax.z));

  return tmin <= tmax;
}

}

#endif // KERNEL_INTERSECTION_HPP