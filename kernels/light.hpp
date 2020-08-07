#ifndef KERNEL_LIGHT_HPP
#define KERNEL_LIGHT_HPP

#include "kernel_types/area_light.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/matrix.hpp"

namespace nova {

// Randomly sample light source uniformly
DEVICE inline float3 sample(const AreaLightData& light, uint& rng_state) {
  float2 offset =
    light.dims * (make_vector<float2>(rand(rng_state), rand(rng_state)) * 2.0f - 1.0f);
  Mat3x3 light_basis = create_basis(normalize(light.normal));
  return light.position + light_basis * make_vector<float3>(offset.x, 0.0f, offset.y);
}

}

#endif // KERNEL_LIGHT_HPP