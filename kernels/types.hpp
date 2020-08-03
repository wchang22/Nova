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

  DEVICE Ray(float3 point, float3 direction, float epsilon)
    : origin(point + direction * epsilon),
      direction(direction),
      inv_direction(1.0f / direction),
      nio(-origin * inv_direction) {}
};

struct Intersection {
  float3 barycentric;
  float length;
  int tri_index;

  DEVICE Intersection() : barycentric(make_vector<float3>(0.0f)), length(FLT_MAX), tri_index(-1) {}
  DEVICE Intersection(float length)
    : barycentric(make_vector<float3>(0.0f)), length(length), tri_index(-1) {}
};

}

#endif // KERNEL_TYPES_HPP