#ifndef CUDA_KERNEL_TEXTURE_H
#define CUDA_KERNEL_TEXTURE_H

#include "kernel_types/triangle.h"

__device__
float3 read_material(cudaTextureObject_t materials, const TriangleMetaData& meta,
                     float2 texture_coord, int index, float3 default_material);

__device__
float3 compute_normal(cudaTextureObject_t materials, const TriangleMetaData& meta,
                      float2 texture_coord, float3 barycentric);

__device__
float3 fresnel_schlick(float cos_theta, float3 f0);

__device__
float distribution_ggx(float n_dot_h, float roughness);

__device__
float geometry_smith(float n_dot_v, float n_dot_l, float nvl, float roughness);

__device__
float3 specularity(float3 view_dir, float3 half_dir, float3 diffuse, float metallic);

__device__
float3 shade(float3 light_dir, float3 view_dir, float3 half_dir, float light_distance, 
             float3 normal, float3 diffuse, float3 kS, float metallic, float roughness);


#endif // CUDA_KERNEL_TEXTURE_H