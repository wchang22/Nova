#ifndef KERNEL_TYPES_HPP
#define KERNEL_TYPES_HPP

#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/backend/vector.hpp"

namespace nova {

struct Ray {
  float3 origin;
  float3 direction;
  float3 inv_direction;
  float3 nio;
};

DEVICE inline Ray create_ray(float3 point, float3 direction, float epsilon) {
  float3 origin = point + direction * epsilon;
  float3 inv_direction = 1.0f / direction;
  float3 nio = -origin * inv_direction;
  return { origin, direction, inv_direction, nio };
}

struct Intersection {
  float3 barycentric;
  float length;
  int tri_index;
};

DEVICE inline Intersection no_intersection() { return { make_vector<float3>(0.0f), FLT_MAX, -1 }; };

}

#endif // KERNEL_TYPES_HPP