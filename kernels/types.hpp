#ifndef KERNEL_TYPES_HPP
#define KERNEL_TYPES_HPP

#include "kernel_types/wavefront.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/backend/vector.hpp"

namespace nova {

struct Ray {
  float3 origin;
  float3 direction;
  float3 inv_direction;
  float3 nio;

  Ray() = default;

  DEVICE Ray(float3 point, float3 direction, float epsilon)
    : origin(point + direction * epsilon),
      direction(direction),
      inv_direction(1.0f / direction),
      nio(-origin * inv_direction) {}

  DEVICE Ray(const PackedRay& ray) : Ray(xyz<float3>(ray.origin_path_index), ray.direction, 0.0f) {}

  DEVICE PackedRay to_packed_ray(uint path_index) {
    return {
      make_vector<float4>(origin, static_cast<float>(path_index)),
      direction,
    };
  }
};

DEVICE uint get_path_index(const PackedRay& ray) { return ray.origin_path_index.w; }

struct Intersection {
  float3 barycentric;
  float length;
  int tri_index;
  int ray_index;

  DEVICE Intersection()
    : barycentric(make_vector<float3>(0.0f)), length(FLT_MAX), tri_index(-1), ray_index(-1) {}
  DEVICE Intersection(float length)
    : barycentric(make_vector<float3>(0.0f)), length(length), tri_index(-1), ray_index(-1) {}

  DEVICE Intersection(const IntersectionData& intrs)
    : barycentric(intrs.barycentric),
      length(intrs.length),
      tri_index(intrs.tri_index),
      ray_index(intrs.ray_index) {}

  DEVICE IntersectionData to_intersection_data() {
    return {
      barycentric,
      length,
      tri_index,
      ray_index,
    };
  }
};

}

#endif // KERNEL_TYPES_HPP