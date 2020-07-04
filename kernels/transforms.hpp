#ifndef KERNEL_TRANSFORMS_HPP
#define KERNEL_TRANSFORMS_HPP

#include "kernels/backend/kernel.hpp"
#include "kernels/backend/vector.hpp"

namespace nova {

DEVICE inline float3 uint3_to_float3(const uint3& u) { return make_vector<float3>(u) / 255.0f; }

DEVICE inline float3 uchar3_to_float3(const uchar3& u) { return make_vector<float3>(u) / 255.0f; }

DEVICE inline uchar3 float3_to_uchar3(const float3& u) {
  return make_vector<uchar3, float3>(u * 255.0f);
}

template <typename U>
DEVICE inline U
triangle_interpolate(const float3& barycentric_coords, const U& a, const U& b, const U& c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

DEVICE inline float3 tone_map(const float3& x, float exposure) { return 1.0f - exp(-x * exposure); }

DEVICE inline float3 gamma_correct(const float3& x) { return pow(x, 1.0f / 2.2f); }

DEVICE inline float rgb_to_luma(const float3& rgb) {
  return dot(rgb, make_vector<float3>(0.299f, 0.587f, 0.114f));
}

}

#endif // KERNEL_TRANSFORMS_HPP
