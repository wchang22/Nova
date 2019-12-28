#ifndef INTERSECTION_CL
#define INTERSECTION_CL

#include "types.cl"

// Use woop transformation to transform ray to unit triangle space
// http://www.sven-woop.de/papers/2004-GH-SaarCOR.pdf
bool intersects_triangle(Ray* ray, int intrs, Triangle tri) {
  float3 o, d;

  o.x = dot(tri.transform_x.xyz, ray->point) + tri.transform_x.w;
  o.y = dot(tri.transform_y.xyz, ray->point) + tri.transform_y.w;
  o.z = dot(tri.transform_z.xyz, ray->point) + tri.transform_z.w;

  d.x = dot(tri.transform_x.xyz, ray->direction);
  d.y = dot(tri.transform_y.xyz, ray->direction);
  d.z = dot(tri.transform_z.xyz, ray->direction);

  float t = -o.z / d.z;
  if (t < 0.0f || t >= ray->length) {
    return false;
  }

  float u = o.x + t * d.x;
  if (u < 0.0f) {
    return false;
  }

  float v = o.y + t * d.y;
  if (v < 0.0f || u + v > 1.0f) {
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
