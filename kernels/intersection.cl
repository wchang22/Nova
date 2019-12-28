#ifndef INTERSECTION_CL
#define INTERSECTION_CL

#include "types.cl"

bool intersects(Ray* ray, int intrs, Triangle tri) {
  float3 normal = cross(tri.edge1, tri.edge2);
  float a = dot(-normal, ray->direction);

  if (a == 0.0) {
    return false;
  }

  float f = 1.0 / a;
  float3 s = ray->point - tri.vertex;
  float t = f * dot(normal, s);

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
bool intersects_aabb(Ray* ray, float3 top, float3 bottom) {
  // Find slab bounds on AABB
  float3 t1 = (top - ray->point) * ray->inv_direction;
  float3 t2 = (bottom - ray->point) * ray->inv_direction;
  float3 tvmin = min(t1, t2);
  float3 tvmax = max(t1, t2);

  // Find tightest components of min and max
  float tmin = max(tvmin.x, max(tvmin.y, tvmin.z));
  float tmax = min(tvmax.x, min(tvmax.y, tvmax.z));

  return tmin <= tmax;
}

#endif // INTERSECTION_CL
