#ifndef INTERSECTION_CL
#define INTERSECTION_CL

#include "types.cl"

bool intersects(Ray* ray, int intersectable_index, Triangle tri) {
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
  ray->intersectable_index = intersectable_index;
  return true;
}

#endif // INTERSECTION_CL