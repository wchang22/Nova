#include "kernels/cuda/entrypoint.hpp"
#include "kernels/kernels.hpp"

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
                     cudaTextureObject_t sky) {
  dim3 block_size { local_dims.x, local_dims.y, 1 };
  dim3 num_blocks { global_dims.x / block_size.x, global_dims.y / block_size.y, 1 };
  kernel_raytrace<<<num_blocks, block_size>>>(scene_params, temp_pixels1, temp_pixels2, pixel_dims,
                                              triangles, tri_meta, bvh, materials, sky);
}

void kernel_interpolate(uint2 global_dims,
                        uint2 local_dims,
                        cudaTextureObject_t temp_pixels1,
                        cudaSurfaceObject_t temp_pixels2,
                        uint2 pixel_dims,
                        uint* rem_pixels_counter,
                        int2* rem_coords) {
  dim3 block_size { local_dims.x, local_dims.y, 1 };
  dim3 num_blocks { global_dims.x / block_size.x, global_dims.y / block_size.y, 1 };
  kernel_interpolate<<<num_blocks, block_size>>>(temp_pixels1, temp_pixels2, pixel_dims,
                                                 rem_pixels_counter, rem_coords);
}

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
                           int2* rem_coords) {
  dim3 block_size { local_dims.x, local_dims.y, 1 };
  dim3 num_blocks { global_dims.x / block_size.x, global_dims.y / block_size.y, 1 };
  kernel_fill_remaining<<<num_blocks, block_size>>>(scene_params, temp_pixels2, pixel_dims,
                                                    triangles, tri_meta, bvh, materials, sky,
                                                    rem_pixels_counter, rem_coords);
}

void kernel_post_process(uint2 global_dims,
                         uint2 local_dims,
                         const SceneParams& scene_params,
                         cudaTextureObject_t temp_pixels2,
                         cudaSurfaceObject_t pixels,
                         uint2 pixel_dims) {
  dim3 block_size { local_dims.x, local_dims.y, 1 };
  dim3 num_blocks { global_dims.x / block_size.x, global_dims.y / block_size.y, 1 };
  kernel_post_process<<<num_blocks, block_size>>>(scene_params, temp_pixels2, pixels, pixel_dims);
}

}