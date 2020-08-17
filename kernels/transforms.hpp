#ifndef KERNEL_TRANSFORMS_HPP
#define KERNEL_TRANSFORMS_HPP

#include "kernels/backend/kernel.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/matrix.hpp"

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

DEVICE inline Mat3x3 create_basis(const float3& normal) {
  float3 v = normal;
  float3 vec =
    fabs(v.y) > 0.1 ? make_vector<float3>(1.0f, 0.0f, 0.0f) : make_vector<float3>(0.0f, 1.0f, 0.0f);
  float3 u = normalize(cross(v, vec));
  float3 w = cross(u, v);

  return { u, v, w };
}

DEVICE inline float2 coords_to_uv(const int2& coords, const uint2& dims) {
  return (make_vector<float2>(coords) + 0.5f) / make_vector<float2>(dims);
}

DEVICE inline float3 spherical_to_cartesian(float theta, float phi) {
  return make_vector<float3>(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

constexpr float NON_ZERO_EPSILON = 1e-7f;

DEVICE inline float make_non_zero(float x) { return max(x, NON_ZERO_EPSILON); }

DEVICE inline float3 make_non_zero(const float3& x) {
  return max(x, make_vector<float3>(NON_ZERO_EPSILON));
}

}

#endif // KERNEL_TRANSFORMS_HPP
