#include "kernels/entrypoint/cuda/entrypoint.hpp"
#include "kernels/kernels.hpp"

namespace nova {

void kernel_generate(dim3 num_blocks,
                     dim3 block_size,
                     // Stage outputs
                     PackedRay* rays,
                     Path* paths,
                     // Stage params
                     SceneParams params,
                     int sample_index,
                     uint time,
                     uint2 pixel_dims) {
  kernel_generate<<<num_blocks, block_size>>>(rays, paths, params, sample_index, time, pixel_dims);
}

void kernel_intersect(dim3 num_blocks,
                      dim3 block_size,
                      // Stage inputs
                      PackedRay* rays,
                      Path* paths,
                      // Stage outputs
                      IntersectionData* intersections,
                      uint* intersection_count,
                      // Stage params
                      uint num_rays,
                      uint denoise_available,
                      TriangleData* triangles,
                      FlatBVHNode* bvh,
                      cudaTextureObject_t sky) {
  kernel_intersect<<<num_blocks, block_size>>>(rays, paths, intersections, intersection_count,
                                               num_rays, denoise_available, triangles, bvh, sky);
}

void kernel_extend(dim3 num_blocks,
                   dim3 block_size,
                   // Stage inputs
                   PackedRay* rays,
                   IntersectionData* intersections,
                   Path* paths,
                   // Stage outputs
                   PackedRay* extended_rays,
                   uint* ray_count,
                   // Stage params
                   uint num_intersections,
                   SceneParams params,
                   uint time,
                   uint denoise_available,
                   TriangleData* triangles,
                   TriangleMetaData* tri_meta,
                   FlatBVHNode* bvh,
                   AreaLightData* lights,
                   uint num_lights,
                   cudaTextureObject_t materials) {
  kernel_extend<<<num_blocks, block_size>>>(
    rays, intersections, paths, extended_rays, ray_count, num_intersections, params, time,
    denoise_available, triangles, tri_meta, bvh, lights, num_lights, materials);
}

void kernel_write(dim3 num_blocks,
                  dim3 block_size,
                  // Stage inputs
                  Path* paths,
                  // Stage outputs
                  cudaSurfaceObject_t temp_color1,
                  cudaSurfaceObject_t albedo_feature1,
                  cudaSurfaceObject_t normal_feature1,
                  // Stage params
                  int sample_index,
                  uint denoise_available,
                  uint2 pixel_dims) {
  kernel_write<<<num_blocks, block_size>>>(paths, temp_color1, albedo_feature1, normal_feature1,
                                           sample_index, denoise_available, pixel_dims);
}

void kernel_raytrace(dim3 num_blocks,
                     dim3 block_size,
                     const SceneParams& scene_params,
                     uint time,
                     cudaSurfaceObject_t temp_color1,
                     uint2 pixel_dims,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     AreaLightData* lights,
                     uint num_lights,
                     cudaTextureObject_t materials,
                     cudaTextureObject_t sky,
                     uint denoise_available,
                     cudaSurfaceObject_t albedo_feature1,
                     cudaSurfaceObject_t normal_feature1) {
  kernel_raytrace<<<num_blocks, block_size>>>(
    scene_params, time, temp_color1, pixel_dims, triangles, tri_meta, bvh, lights, num_lights,
    materials, sky, denoise_available, albedo_feature1, normal_feature1);
}

void kernel_accumulate(dim3 num_blocks,
                       dim3 block_size,
                       int sample_index,
                       uint denoise_available,
                       cudaTextureObject_t temp_color1,
                       cudaTextureObject_t albedo_feature1,
                       cudaTextureObject_t normal_feature1,
                       cudaTextureObject_t prev_color,
                       cudaTextureObject_t prev_albedo_feature,
                       cudaTextureObject_t prev_normal_feature,
                       cudaSurfaceObject_t temp_color2,
                       cudaSurfaceObject_t albedo_feature2,
                       cudaSurfaceObject_t normal_feature2,
                       uint2 pixel_dims) {
  kernel_accumulate<<<num_blocks, block_size>>>(
    sample_index, denoise_available, temp_color1, albedo_feature1, normal_feature1, prev_color,
    prev_albedo_feature, prev_normal_feature, temp_color2, albedo_feature2, normal_feature2,
    pixel_dims);
}

void kernel_post_process(dim3 num_blocks,
                         dim3 block_size,
                         const SceneParams& scene_params,
                         cudaTextureObject_t temp_color2,
                         cudaSurfaceObject_t pixels,
                         uint2 pixel_dims) {
  kernel_post_process<<<num_blocks, block_size>>>(scene_params, temp_color2, pixels, pixel_dims);
}

}