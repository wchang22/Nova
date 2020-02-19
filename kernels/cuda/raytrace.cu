#include "raytrace.h"

void kernel_raytrace(std::tuple<uint32_t, uint32_t, uint32_t> global_size,
                     std::tuple<uint32_t, uint32_t, uint32_t> local_size,
                     cudaTextureObject_t image_out,
                     EyeCoords ec,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials) {
  
}