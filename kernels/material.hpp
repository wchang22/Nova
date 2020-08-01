#ifndef KERNEL_MATERIAL_HPP
#define KERNEL_MATERIAL_HPP

#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"
#include "kernels/backend/image.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/matrix.hpp"
#include "kernels/transforms.hpp"

namespace nova {

DEVICE inline float3 read_material(image2d_array_read_t materials,
                                   const TriangleMetaData& meta,
                                   const float2& texture_coord,
                                   int index,
                                   const float3& default_material) {
  if (meta.diffuse_index == -1 && meta.metallic_index == -1 && meta.roughness_index == -1 &&
      meta.ambient_occlusion_index == -1 && meta.normal_index == -1) {
    return default_material;
  }
  if (index == -1) {
    return default_material;
  }

  return xyz<float3>(read_image<float4, AddressMode::WRAP>(materials, texture_coord, index));
}

DEVICE inline float3 read_sky(image2d_read_t sky, float3 direction) {
  float2 uv = make_vector<float2>(atan2(direction.z, direction.x), asin(direction.y));
  uv = uv * make_vector<float2>(M_1_PI_F * 0.5f, M_1_PI_F) + 0.5f;
  return xyz<float3>(read_image<float4, AddressMode::WRAP>(sky, uv));
}

DEVICE inline float3 compute_normal(image2d_array_read_t materials,
                                    const TriangleMetaData& meta,
                                    const float2& texture_coord,
                                    const float3& barycentric) {
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

}

#endif // KERNEL_MATERIAL_HPP