#ifndef CUDA_KERNEL_RAYTRACE_H
#define CUDA_KERNEL_RAYTRACE_H

#include "kernel_types/kernel_constants.h"
#include "kernel_types/eye_coords.h"
#include "kernel_types/triangle.h"
#include "kernel_types/bvh_node.h"

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

#endif // CUDA_KERNEL_RAYTRACE_H