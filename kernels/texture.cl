#ifndef TEXTURE_CL
#define TEXTURE_CL

constant sampler_t material_sampler =
  CLK_ADDRESS_CLAMP |
  CLK_FILTER_NEAREST |
  CLK_NORMALIZED_COORDS_TRUE;

float3 read_material(read_only image2d_array_t materials, TriangleMeta meta,
                     float2 texture_coord, int index, float3 default_material) {
  if (meta.ambient_index == -1 && meta.diffuse_index == -1 && meta.specular_index == -1 &&
      meta.normal_index == -1) {
    return default_material;
  }
  if (index == -1) {
    return 0;
  }

  float3 texture = convert_float3(
    read_imageui(materials, material_sampler, (float4)(texture_coord, index, 0)).xyz
  );

  return uint3_to_float3(texture);
}

float3 compute_normal(read_only image2d_array_t materials, TriangleMeta meta,
                      float2 texture_coord, float3 barycentric) {
  // Interpolate triangle normal from vertex data
  float3 normal = fast_normalize(
    triangle_interpolate3(barycentric, meta.normal1, meta.normal2, meta.normal3)
  );

  // Use the normal map to compute pixel normal if it exists
  if (meta.normal_index != -1) {
    float3 tangent = fast_normalize(
      triangle_interpolate3(barycentric, meta.tangent1, meta.tangent2, meta.tangent3)
    );
    float3 bitangent = fast_normalize(
      triangle_interpolate3(barycentric, meta.bitangent1, meta.bitangent2, meta.bitangent3)
    );

    // Create TBN matrix and use it to convert tangent space pixel normal to world space
    Mat3x3 tbn = mat3x3_transpose((Mat3x3) {
      tangent,
      bitangent,
      normal
    });

    float3 pixel_normal = read_material(materials, meta, texture_coord, meta.normal_index, 0);
    pixel_normal = fast_normalize(pixel_normal * 2.0f - 1.0f);
    normal = fast_normalize(mat3x3_vec3_mult(tbn, pixel_normal));
  }

  return normal;
}

float3 shade(float3 light_dir, float3 eye_dir, float3 normal,
             float3 diffuse, float3 specular, int shininess) {
  float3 half_dir = fast_normalize(light_dir - eye_dir);
  float3 diffuse_shading = diffuse * max(dot(normal, light_dir), 0.f);
  float3 specular_shading = specular * pown(max(dot(normal, half_dir), 0.f), shininess);

  return diffuse_shading + specular_shading;
}

#endif // TEXTURE_CL
