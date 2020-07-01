#ifndef KERNEL_TEXTURE_HPP
#define KERNEL_TEXTURE_HPP

#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/new/matrix.hpp"
#include "kernels/new/transforms.hpp"

// #define cudaTextureObject_t float

// template <typename T>
// DEVICE T tex2DLayered(cudaTextureObject_t a, float x, float y, int i) {
//   return nova::make_vector<T>(0.0f);
// }
// template <typename T>
// DEVICE T tex2D(cudaTextureObject_t a, float x, float y) {
//   return nova::make_vector<T>(0.0f);
// }

namespace nova {

DEVICE inline float3 read_material(cudaTextureObject_t materials,
                                   const TriangleMetaData& meta,
                                   float2 texture_coord,
                                   int index,
                                   float3 default_material) {
  if (meta.diffuse_index == -1 && meta.metallic_index == -1 && meta.roughness_index == -1 &&
      meta.ambient_occlusion_index == -1 && meta.normal_index == -1) {
    return default_material;
  }
  if (index == -1) {
    return default_material;
  }

  return xyz<float3>(tex2DLayered<float4>(materials, texture_coord.x, texture_coord.y, index));
}

DEVICE inline float3 read_sky(cudaTextureObject_t sky, float3 direction) {
  float2 uv = make_vector<float2>(atan2(direction.z, direction.x), asin(direction.y));
  uv = uv * make_vector<float2>(M_1_PI_F * 0.5f, M_1_PI_F) + 0.5f;
  return xyz<float3>(tex2D<float4>(sky, uv.x, uv.y));
}

DEVICE inline float3 compute_normal(cudaTextureObject_t materials,
                                    const TriangleMetaData& meta,
                                    float2 texture_coord,
                                    float3 barycentric) {
  // Interpolate triangle normal from vertex data
  float3 normal =
    normalize(triangle_interpolate(barycentric, meta.normal1, meta.normal2, meta.normal3));

  // Use the normal map to compute pixel normal if it exists
  if (meta.normal_index != -1) {
    float3 tangent =
      normalize(triangle_interpolate(barycentric, meta.tangent1, meta.tangent2, meta.tangent3));
    float3 bitangent = normalize(
      triangle_interpolate(barycentric, meta.bitangent1, meta.bitangent2, meta.bitangent3));

    // Create TBN matrix and use it to convert tangent space pixel normal to world space
    Mat3x3 tbn = transpose({ tangent, bitangent, normal });

    float3 pixel_normal =
      read_material(materials, meta, texture_coord, meta.normal_index, make_vector<float3>(0.0f));
    pixel_normal = normalize(pixel_normal * 2.0f - 1.0f);
    normal = normalize(tbn * pixel_normal);
  }

  return normal;
}

DEVICE inline float3 fresnel_schlick(float cos_theta, float3 f0) {
  float a = 1.0f - cos_theta;
  float a2 = a * a;
  float a5 = a2 * a2 * a;
  return f0 + (1.0f - f0) * a5;
}

DEVICE inline float distribution_ggx(float n_dot_h, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;

  float denom = n_dot_h * n_dot_h * (a2 - 1.0f) + 1.0f;
  denom = M_PI * denom * denom;

  return a2 / denom;
}

DEVICE inline float geometry_smith(float n_dot_v, float n_dot_l, float nvl, float roughness) {
  float r = roughness + 1.0f;
  float k = r * r / 8.0f;
  float m = 1.0f - k;

  return nvl / ((n_dot_v * m + k) * (n_dot_l * m + k));
}

DEVICE inline float3 specularity(float3 view_dir, float3 half_dir, float3 diffuse, float metallic) {
  float h_dot_v = max(dot(half_dir, view_dir), 0.0f);
  float3 f0 = mix(make_vector<float3>(0.04f), diffuse, metallic);
  // fresnel equation
  float3 f = fresnel_schlick(h_dot_v, f0);

  return f;
}

DEVICE inline float3 shade(const SceneParams& params,
                           float3 light_dir,
                           float3 view_dir,
                           float3 half_dir,
                           float light_distance,
                           float3 normal,
                           float3 diffuse,
                           float3 kS,
                           float metallic,
                           float roughness) {
  float n_dot_l = max(dot(normal, light_dir), 0.0f);
  if (n_dot_l == 0.0f) {
    return make_vector<float3>(0.0f);
  }
  float n_dot_v = max(dot(normal, view_dir), 0.0f);
  float n_dot_h = max(dot(normal, half_dir), 0.0f);

  float nvl = n_dot_v * n_dot_l;

  // normal distribution function
  float d = distribution_ggx(n_dot_h, roughness);
  // geometry function
  float g = geometry_smith(n_dot_v, n_dot_l, nvl, roughness);

  // diffuse
  float3 kD = (1.0f - kS) * (1.0f - metallic);

  float3 brdf = kD * diffuse * M_1_PI_F + d * kS * g / max(4.0f * nvl, 1e-3f);
  float3 radiance = params.light_intensity / max(light_distance * light_distance, 1.0f);

  return brdf * radiance * n_dot_l;
}

}

#endif // KERNEL_TEXTURE_HPP