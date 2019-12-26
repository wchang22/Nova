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

  float t = -native_divide(woop_ray.origin.z, woop_ray.direction.z);
  if (t < 0.0f || t >= intrs->length) {
    return false;
  }

  float2 uv = woop_ray.origin.xy + t * woop_ray.direction.xy;
  float3 barycentric = (float3)(1.0f - uv.x - uv.y, uv);

  if (any(isless(barycentric, 0.0f))) {
    return false;
  }

  intrs->length = t;
  intrs->barycentric = barycentric;
  intrs->tri_index = tri_index;
  return true;
}

// AABB fast intersection for BVH
bool intersects_aabb(Ray ray, float3 top, float3 bottom) {
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

#endif // INTERSECTION_CL