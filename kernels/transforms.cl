#ifndef TRANSFORMS_CL
#define TRANSFORMS_CL

#include "types.cl"

float3 reflect(float3 I, float3 N) {
  return I - 2.0f * dot(I, N) * N;
}

// Transform ray to unit triangle space
Ray transform_ray(Ray ray, Triangle tri) {
  Ray woop_ray = ray;

  woop_ray.point.x = dot(tri.transform_x.xyz, ray.point) + tri.transform_x.w;
  woop_ray.point.y = dot(tri.transform_y.xyz, ray.point) + tri.transform_y.w;
  woop_ray.point.z = dot(tri.transform_z.xyz, ray.point) + tri.transform_z.w;

  woop_ray.direction.x = dot(tri.transform_x.xyz, ray.direction);
  woop_ray.direction.y = dot(tri.transform_y.xyz, ray.direction);
  woop_ray.direction.z = dot(tri.transform_z.xyz, ray.direction);

  return woop_ray;
}

float3 triangle_interpolate(float3 barycentric_coords, float3 a, float3 b, float3 c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

#endif // TRANSFORMS_CL
