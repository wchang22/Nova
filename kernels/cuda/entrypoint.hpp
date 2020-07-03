#ifndef KERNEL_CUDA_ENTRYPOINT_HPP
#define KERNEL_CUDA_ENTRYPOINT_HPP

#include "kernel_types/bvh_node.hpp"
#include "kernel_types/eye_coords.hpp"
#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"

namespace nova {

void kernel_raytrace(uint2 global_dims,
                     uint2 local_dims,
                     const SceneParams& scene_params,
                     cudaSurfaceObject_t temp_pixels1,
                     cudaSurfaceObject_t temp_pixels2,
                     uint2 pixel_dims,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials,
                     cudaTextureObject_t sky);

void kernel_interpolate(uint2 global_dims,
                        uint2 local_dims,
                        cudaTextureObject_t temp_pixels1,
                        cudaSurfaceObject_t temp_pixels2,
                        uint2 pixel_dims,
                        uint* rem_pixels_counter,
                        int2* rem_coords);

void kernel_fill_remaining(uint2 global_dims,
                           uint2 local_dims,
                           const SceneParams& scene_params,
                           cudaSurfaceObject_t temp_pixels2,
                           uint2 pixel_dims,
                           TriangleData* triangles,
                           TriangleMetaData* tri_meta,
                           FlatBVHNode* bvh,
                           cudaTextureObject_t materials,
                           cudaTextureObject_t sky,
                           uint* rem_pixels_counter,
                           int2* rem_coords);

void kernel_post_process(uint2 global_dims,
                         uint2 local_dims,
                         const SceneParams& scene_params,
                         cudaTextureObject_t temp_pixels2,
                         cudaSurfaceObject_t pixels,
                         uint2 pixel_dims);

}

#endif // CUDA_KERNEL_RAYTRACE_HPP