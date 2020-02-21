#include "raytrace.h"
#include "constants.h"
#include "vector_math.h"
#include "vector_conversions.h"

__global__
void raytrace(cudaSurfaceObject_t image_out,
              EyeCoords ec,
              TriangleData* triangles,
              TriangleMetaData* tri_meta,
              FlatBVHNode* bvh,
              cudaTextureObject_t materials) {
  uint2 pixel_coords = { blockDim.x * blockIdx.x + threadIdx.x,
                         blockDim.y * blockIdx.y + threadIdx.y };
  float3 color = { 0.0f, 0.0f, 0.0f };
  


  uchar4 image_color = convert_uchar4(convert_float4(color, 1.0f) * 255.0f);
  surf2Dwrite(image_color, image_out, pixel_coords.x * sizeof(uchar4), pixel_coords.y);
}

void kernel_raytrace(uint3 global_dims,
                     const KernelConstants& kernel_constants,
                     cudaSurfaceObject_t image_out,
                     EyeCoords ec,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials) {
  dim3 num_blocks = global_dims;
  dim3 block_size { 1, 1, 1 };
  constants = kernel_constants;
  raytrace<<<num_blocks, block_size>>>(image_out, ec, triangles, tri_meta, bvh, materials);
}