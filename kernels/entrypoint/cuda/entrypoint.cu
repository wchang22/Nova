#include "kernels/entrypoint/cuda/entrypoint.hpp"
#include "kernels/kernels.hpp"

namespace nova {

void kernel_raytrace(dim3 num_blocks,
                     dim3 block_size,
                     const SceneParams& scene_params,
                     uint time,
                     cudaSurfaceObject_t temp_pixels1,
                     cudaSurfaceObject_t temp_pixels2,
                     uint2 pixel_dims,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials,
                     cudaTextureObject_t sky) {
  kernel_raytrace<<<num_blocks, block_size>>>(scene_params, time, temp_pixels1, temp_pixels2,
                                              pixel_dims, triangles, tri_meta, bvh, materials, sky);
}

void kernel_interpolate(dim3 num_blocks,
                        dim3 block_size,
                        cudaTextureObject_t temp_pixels1,
                        cudaSurfaceObject_t temp_pixels2,
                        uint2 pixel_dims,
                        uint* rem_pixels_counter,
                        int2* rem_coords) {
  kernel_interpolate<<<num_blocks, block_size>>>(temp_pixels1, temp_pixels2, pixel_dims,
                                                 rem_pixels_counter, rem_coords);
}

void kernel_fill_remaining(dim3 num_blocks,
                           dim3 block_size,
                           const SceneParams& scene_params,
                           uint time,
                           cudaSurfaceObject_t temp_pixels2,
                           uint2 pixel_dims,
                           TriangleData* triangles,
                           TriangleMetaData* tri_meta,
                           FlatBVHNode* bvh,
                           cudaTextureObject_t materials,
                           cudaTextureObject_t sky,
                           uint* rem_pixels_counter,
                           int2* rem_coords) {
  kernel_fill_remaining<<<num_blocks, block_size>>>(scene_params, time, temp_pixels2, pixel_dims,
                                                    triangles, tri_meta, bvh, materials, sky,
                                                    rem_pixels_counter, rem_coords);
}

void kernel_accumulate(dim3 num_blocks,
                       dim3 block_size,
                       int sample_num,
                       cudaTextureObject_t temp_pixels2,
                       cudaTextureObject_t prev_pixels,
                       cudaSurfaceObject_t temp_pixels1,
                       uint2 pixel_dims) {
  kernel_accumulate<<<num_blocks, block_size>>>(sample_num, temp_pixels2, prev_pixels, temp_pixels1,
                                                pixel_dims);
}

void kernel_post_process(dim3 num_blocks,
                         dim3 block_size,
                         const SceneParams& scene_params,
                         cudaTextureObject_t temp_pixels1,
                         cudaSurfaceObject_t pixels,
                         uint2 pixel_dims) {
  kernel_post_process<<<num_blocks, block_size>>>(scene_params, temp_pixels1, pixels, pixel_dims);
}

}