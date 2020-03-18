#ifndef CUDA_KERNEL_RAYTRACE_HPP
#define CUDA_KERNEL_RAYTRACE_HPP

#include "kernel_types/kernel_constants.hpp"
#include "kernel_types/eye_coords.hpp"
#include "kernel_types/triangle.hpp"
#include "kernel_types/bvh_node.hpp"

void kernel_raytrace(uint3 global_dims,
                     const KernelConstants& kernel_constants,
                     uchar4* pixels,
                     uint2 pixel_dims,
                     EyeCoords ec,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials);

void kernel_interpolate(uint3 global_dims,
                        const KernelConstants& kernel_constants,
                        uchar4* pixels,
                        uint2 pixel_dims,
                        EyeCoords ec,
                        TriangleData* triangles,
                        TriangleMetaData* tri_meta,
                        FlatBVHNode* bvh,
                        cudaTextureObject_t materials);

#endif // CUDA_KERNEL_RAYTRACE_HPP