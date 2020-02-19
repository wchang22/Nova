#ifndef CUDA_KERNEL_RAYTRACE_H
#define CUDA_KERNEL_RAYTRACE_H

#include <tuple>

#include "matrix.h"
#include "kernel_types.h"

void kernel_raytrace(std::tuple<uint32_t, uint32_t, uint32_t> global_size,
                     std::tuple<uint32_t, uint32_t, uint32_t> local_size,
                     cudaTextureObject_t image_out,
                     EyeCoords ec,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials);

#endif // CUDA_KERNEL_RAYTRACE_H