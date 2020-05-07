#ifndef TRANSFORMS_CL
#define TRANSFORMS_CL

#include "types.cl"

float3 reflect(float3 I, float3 N) { return I - 2.0f * dot(I, N) * N; }

float3 uint3_to_float3(float3 u) { return u / 255.0f; }

float3 triangle_interpolate3(float3 barycentric_coords, float3 a, float3 b, float3 c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

float2 triangle_interpolate2(float3 barycentric_coords, float2 a, float2 b, float2 c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

int linear_index(int2 p, int width) { return p.y * width + p.x; }

float3 tone_map(float3 x, float exposure) { return 1.0f - exp(-x * exposure); }

float3 gamma_correct(float3 x) { return pow(x, 1.0f / 2.2f); }

#endif // TRANSFORMS_CL
