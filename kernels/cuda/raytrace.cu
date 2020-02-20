#include "raytrace.h"

void kernel_raytrace(const Dims& global_dims,
                     const KernelConstants& kernel_constants,
                     cudaTextureObject_t image_out,
                     EyeCoords ec,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials) {
  
}