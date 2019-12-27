#ifndef SHADING_CL
#define SHADING_CL

float3 shade(float3 light_dir, float3 eye_dir, float3 normal,
             float3 diffuse, float3 specular, int shininess) {
  float3 half_dir = fast_normalize(light_dir - eye_dir);
  float3 diffuse_shading = diffuse * max(dot(normal, light_dir), 0.f);
  float3 specular_shading = specular * pown(max(dot(normal, half_dir), 0.f), shininess);

  return diffuse_shading + specular_shading;
}

#endif // SHADING_CL
