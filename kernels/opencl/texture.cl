#ifndef TEXTURE_CL
#define TEXTURE_CL

constant sampler_t material_sampler =
  CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;

float3 read_material(read_only image2d_array_t materials,
                     TriangleMeta meta,
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

  return read_imagef(materials, material_sampler, (float4)(texture_coord, index, 0.0f)).xyz;
}

float3 compute_normal(read_only image2d_array_t materials,
                      TriangleMeta meta,
                      float2 texture_coord,
                      float3 barycentric) {
  // Interpolate triangle normal from vertex data
  float3 normal =
    fast_normalize(triangle_interpolate3(barycentric, meta.normal1, meta.normal2, meta.normal3));

  // Use the normal map to compute pixel normal if it exists
  if (meta.normal_index != -1) {
    float3 tangent = fast_normalize(
      triangle_interpolate3(barycentric, meta.tangent1, meta.tangent2, meta.tangent3));
    float3 bitangent = fast_normalize(
      triangle_interpolate3(barycentric, meta.bitangent1, meta.bitangent2, meta.bitangent3));

    // Create TBN matrix and use it to convert tangent space pixel normal to world space
    Mat3x3 tbn = mat3x3_transpose((Mat3x3) { tangent, bitangent, normal });

    float3 pixel_normal = read_material(materials, meta, texture_coord, meta.normal_index, 0);
    pixel_normal = fast_normalize(pixel_normal * 2.0f - 1.0f);
    normal = fast_normalize(mat3x3_vec3_mult(tbn, pixel_normal));
  }

  return normal;
}

float3 fresnel_schlick(float cos_theta, float3 f0) {
  return f0 + (1.0f - f0) * pown(1.0f - cos_theta, 5);
}

float distribution_ggx(float n_dot_h, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;

  float denom = n_dot_h * n_dot_h * (a2 - 1.0f) + 1.0f;
  denom = M_PI_F * denom * denom;

  return native_divide(a2, denom);
}

float geometry_smith(float n_dot_v, float n_dot_l, float nvl, float roughness) {
  float r = roughness + 1.0f;
  float k = native_divide(r * r, 8.0f);
  float m = 1.0f - k;

  return native_divide(nvl, (n_dot_v * m + k) * (n_dot_l * m + k));
}

float3 specularity(float3 view_dir, float3 half_dir, float3 diffuse, float metallic) {
  float h_dot_v = max(dot(half_dir, view_dir), 0.0f);
  float3 f0 = mix(0.04f, diffuse, metallic);
  // fresnel equation
  float3 f = fresnel_schlick(h_dot_v, f0);

  return f;
}

float3 shade(SceneParams scene_params,
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
    return 0.0f;
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

  float3 brdf = kD * diffuse * M_1_PI_F + native_divide(d * kS * g, max(4.0f * nvl, 1e-3f));
  float3 radiance =
    native_divide(scene_params.light_intensity, max(light_distance * light_distance, 1.0f));

  return brdf * radiance * n_dot_l;
}

#endif // TEXTURE_CL
