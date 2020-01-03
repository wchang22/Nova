#ifndef SHADING_CL
#define SHADING_CL

constant sampler_t material_sampler =
  CLK_ADDRESS_CLAMP |
  CLK_FILTER_NEAREST |
  CLK_NORMALIZED_COORDS_TRUE;

float3 read_material(read_only image2d_array_t materials, TriangleMeta meta,
                     float2 texture_coord, int index, float3 default_material) {
  if (!meta.has_textures) {
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

float3 shade(float3 light_dir, float3 eye_dir, float3 normal,
             float3 diffuse, float3 specular, int shininess) {
  float3 half_dir = fast_normalize(light_dir - eye_dir);
  float3 diffuse_shading = diffuse * max(dot(normal, light_dir), 0.f);
  float3 specular_shading = specular * pown(max(dot(normal, half_dir), 0.f), shininess);

  return diffuse_shading + specular_shading;
}

#endif // SHADING_CL
