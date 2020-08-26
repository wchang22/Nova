#ifndef KERNEL_CUDA_ENTRYPOINT_HPP
#define KERNEL_CUDA_ENTRYPOINT_HPP

#include "kernel_types/area_light.hpp"
#include "kernel_types/bvh_node.hpp"
#include "kernel_types/eye_coords.hpp"
#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"

namespace nova {

void kernel_raytrace(dim3 num_blocks,
                     dim3 block_size,
                     const SceneParams& scene_params,
                     uint time,
                     cudaSurfaceObject_t temp_color1,
                     uint2 pixel_dims,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     AreaLightData* lights,
                     uint num_lights,
                     cudaTextureObject_t materials,
                     cudaTextureObject_t sky,
                     uint denoise_available,
                     cudaSurfaceObject_t albedo_feature1,
                     cudaSurfaceObject_t normal_feature1);

void kernel_accumulate(dim3 num_blocks,
                       dim3 block_size,
                       int sample_index,
                       uint denoise_available,
                       cudaTextureObject_t temp_color1,
                       cudaTextureObject_t albedo_feature1,
                       cudaTextureObject_t normal_feature1,
                       cudaTextureObject_t prev_color,
                       cudaTextureObject_t prev_albedo_feature,
                       cudaTextureObject_t prev_normal_feature,
                       cudaSurfaceObject_t temp_color2,
                       cudaSurfaceObject_t albedo_feature2,
                       cudaSurfaceObject_t normal_feature2,
                       uint2 pixel_dims);

void kernel_post_process(dim3 num_blocks,
                         dim3 block_size,
                         const SceneParams& scene_params,
                         cudaTextureObject_t temp_color2,
                         cudaSurfaceObject_t pixels,
                         uint2 pixel_dims);

}

#endif // CUDA_KERNEL_RAYTRACE_HPP