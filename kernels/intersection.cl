#ifndef INTERSECTION_CL
#define INTERSECTION_CL

#include "types.cl"
#include "transforms.cl"

// Use woop transformation to transform ray to unit triangle space
// http://www.sven-woop.de/papers/2004-GH-SaarCOR.pdf
bool intersects_triangle(Ray ray, Intersection* intrs, int tri_index, Triangle tri) {
  // Transform ray to unit triangle space
  Ray woop_ray = ray;
  woop_ray.origin = mat4x3_vec3_mult(tri.transform, ray.origin);
  woop_ray.direction = mat3x3_vec3_mult(mat4x3_to_mat3x3(tri.transform), ray.direction);

  float t = -woop_ray.origin.z / woop_ray.direction.z;
  if (t < 0.0f || t >= intrs->length) {
    return false;
  }

  float u = woop_ray.origin.x + t * woop_ray.direction.x;
  if (u < 0.0f) {
    return false;
  }

  float v = woop_ray.origin.y + t * woop_ray.direction.y;
  if (v < 0.0f || u + v > 1.0f) {
    return false;
  }

  intrs->length = t;
  intrs->barycentric = (float3)(1.0f - u - v, u, v);
  intrs->tri_index = tri_index;
  return true;
}

// AABB fast intersection for BVH
bool intersects_aabb(Ray ray, float3 top, float3 bottom) {
  float3 inv_direction = 1.0f / ray.direction;

  // Find slab bounds on AABB
  float3 t1 = (top - ray.origin) * inv_direction;
  float3 t2 = (bottom - ray.origin) * inv_direction;
  float3 tvmin = min(t1, t2);
  float3 tvmax = max(t1, t2);

  // Find tightest components of min and max
  float tmin = max(tvmin.x, max(tvmin.y, tvmin.z));
  float tmax = min(tvmax.x, min(tvmax.y, tvmax.z));

  return tmin <= tmax;
}

#endif // INTERSECTION_CL
