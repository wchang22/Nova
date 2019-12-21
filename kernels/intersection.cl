#ifndef INTERSECTION_CL
#define INTERSECTION_CL

#include "types.cl"

bool intersects(Ray* ray, int intrs, Triangle tri) {
  float a = dot(-tri.normal, ray->direction);

  float f = 1.0 / a;
  float3 s = ray->point - tri.vertex;
  float t = f * dot(tri.normal, s);

  if (t < 0.0 || t >= ray->length) {
    return false;
  }

  float3 m = cross(s, ray->direction);
  float u = f * dot(m, tri.edge2);

  if (u < 0.0) {
    return false;
  }

  float v = f * dot(-m, tri.edge1);

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  ray->length = t;
  ray->intrs = intrs;
  return true;
}

// AABB fast intersection for BVH
bool intersects_aabb(Ray* ray, AABB aabb) {
  // Find slab bounds on AABB
  float3 t1 = (aabb.top - ray->point) * ray->inv_direction;
  float3 t2 = (aabb.bottom - ray->point) * ray->inv_direction;
  float3 tvmin = min(t1, t2);
  float3 tvmax = max(t1, t2);

  // Find tightest components of min and max
  float tmin = max(tvmin.x, max(tvmin.y, tvmin.z));
  float tmax = min(tvmax.x, min(tvmax.y, tvmax.z));

  return tmin <= tmax;
}

#endif // INTERSECTION_CL