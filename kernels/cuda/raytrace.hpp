#ifndef CUDA_KERNEL_RAYTRACE_HPP
#define CUDA_KERNEL_RAYTRACE_HPP

#include "kernel_types/bvh_node.hpp"
#include "kernel_types/eye_coords.hpp"
#include "kernel_types/kernel_constants.hpp"
#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"

namespace nova {

void kernel_raytrace(uint2 global_dims,
                     uint2 local_dims,
                     const KernelConstants& kernel_constants,
                     const SceneParams& scene_params,
                     uchar4* pixels,
                     uint2 pixel_dims,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials,
                     cudaTextureObject_t sky);

void kernel_interpolate(uint2 global_dims,
                        uint2 local_dims,
                        const KernelConstants& kernel_constants,
                        uchar4* pixels,
                        uint2 pixel_dims,
                        uint* rem_pixels_counter,
                        uint2* rem_coords);

void kernel_fill_remaining(uint2 global_dims,
                           uint2 local_dims,
                           const KernelConstants& kernel_constants,
                           const SceneParams& scene_params,
                           uchar4* pixels,
                           uint2 pixel_dims,
                           TriangleData* triangles,
                           TriangleMetaData* tri_meta,
                           FlatBVHNode* bvh,
                           cudaTextureObject_t materials,
                           cudaTextureObject_t sky,
                           uint* rem_pixels_counter,
                           uint2* rem_coords);

}

#endif // CUDA_KERNEL_RAYTRACE_HPP