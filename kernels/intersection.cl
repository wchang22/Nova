#ifndef INTERSECTION_CL
#define INTERSECTION_CL

#include "types.cl"
#include "transforms.cl"

// Use woop transformation to transform ray to unit triangle space
// http://www.sven-woop.de/papers/2004-GH-SaarCOR.pdf
bool intersects_triangle(Ray* ray, int intrs, Triangle tri) {
  Ray woop_ray = transform_ray(*ray, tri);

  float t = -woop_ray.point.z / woop_ray.direction.z;
  if (t < 0.0f || t >= ray->length) {
    return false;
  }

  float u = woop_ray.point.x + t * woop_ray.direction.x;
  if (u < 0.0f) {
    return false;
  }

  float v = woop_ray.point.y + t * woop_ray.direction.y;
  if (v < 0.0f || u + v > 1.0f) {
    return false;
  }

  ray->length = t;
  ray->intrs = intrs;
  ray->barycentric_coords.x = 1.0f - u - v;
  ray->barycentric_coords.y = u;
  ray->barycentric_coords.z = v;
  return true;
}

// AABB fast intersection for BVH
bool intersects_aabb(Ray* ray, float3 top, float3 bottom) {
  float3 inv_direction = 1.0f / ray->direction;

  // Find slab bounds on AABB
  float3 t1 = (top - ray->point) * inv_direction;
  float3 t2 = (bottom - ray->point) * inv_direction;
  float3 tvmin = min(t1, t2);
  float3 tvmax = max(t1, t2);

  // Find tightest components of min and max
  float tmin = max(tvmin.x, max(tvmin.y, tvmin.z));
  float tmax = min(tvmax.x, min(tvmax.y, tvmax.z));

  return tmin <= tmax;
}

#endif // INTERSECTION_CL
