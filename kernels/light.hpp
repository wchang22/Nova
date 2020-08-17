#ifndef KERNEL_LIGHT_HPP
#define KERNEL_LIGHT_HPP

#include "kernel_types/area_light.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/matrix.hpp"

namespace nova {

// Randomly sample light source uniformly
DEVICE inline float3 sample(const AreaLightData& light, uint& rng_state) {
  if (light.type == AreaLightType::RECT) {
    float2 offset = xy<float2>(light.rect.dims) *
                    (make_vector<float2>(rand(rng_state), rand(rng_state)) * 2.0f - 1.0f);
    Mat3x3 light_basis = create_basis(normalize(light.rect.normal));
    return light.rect.position + light_basis * make_vector<float3>(offset.x, 0.0f, offset.y);
  }

  // Generate random point in triangle: https://www.cs.princeton.edu/~funk/tog02.pdf 4.2
  float r1 = sqrt(rand(rng_state));
  float r2 = rand(rng_state);
  return (1.0f - r1) * light.tri.v1 + r1 * (1.0f - r2) * light.tri.v2 + r1 * r2 * light.tri.v3;
}

DEVICE inline float3 compute_normal(const AreaLightData& light) {
  if (light.type == AreaLightType::RECT) {
    return normalize(light.rect.normal);
  }

  float3 e1 = light.tri.v2 - light.tri.v1;
  float3 e2 = light.tri.v3 - light.tri.v1;
  return normalize(cross(e1, e2));
}

DEVICE inline float compute_area(const AreaLightData& light) {
  if (light.type == AreaLightType::RECT) {
    return light.rect.dims.x * light.rect.dims.y;
  }

  float3 e1 = light.tri.v2 - light.tri.v1;
  float3 e2 = light.tri.v3 - light.tri.v1;
  return 0.5f * length(cross(e1, e2));
}

}

#endif // KERNEL_LIGHT_HPP