#ifndef KERNEL_KERNELS_HPP
#define KERNEL_KERNELS_HPP

#include "kernel_types/area_light.hpp"
#include "kernel_types/bvh_node.hpp"
#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"
#include "kernels/backend/image.hpp"
#include "kernels/backend/kernel.hpp"

namespace nova {

KERNEL void kernel_generate(
  // Stage outputs
  GLOBAL PackedRay* rays,
  GLOBAL Path* paths,
  // Stage params
  SceneParams params,
  int sample_index,
  uint time,
  uint2 pixel_dims);

KERNEL void kernel_intersect(
  // Stage inputs
  GLOBAL PackedRay* rays,
  GLOBAL Path* paths,
  // Stage outputs
  GLOBAL IntersectionData* intersections,
  GLOBAL uint* intersection_count,
  // Stage params
  uint num_rays,
  uint denoise_available,
  GLOBAL TriangleData* triangles,
  GLOBAL FlatBVHNode* bvh,
  image2d_read_t sky);

KERNEL void kernel_extend(
  // Stage inputs
  GLOBAL PackedRay* rays,
  GLOBAL IntersectionData* intersections,
  GLOBAL Path* paths,
  // Stage outputs
  GLOBAL PackedRay* extended_rays,
  GLOBAL uint* ray_count,
  // Stage params
  uint num_intersections,
  SceneParams params,
  uint time,
  uint denoise_available,
  GLOBAL TriangleData* triangles,
  GLOBAL TriangleMetaData* tri_meta,
  GLOBAL FlatBVHNode* bvh,
  GLOBAL AreaLightData* lights,
  uint num_lights,
  image2d_array_read_t materials);

KERNEL void kernel_write(
  // Stage inputs
  GLOBAL Path* paths,
  // Stage outputs
  image2d_write_t temp_color1,
  image2d_write_t albedo_feature1,
  image2d_write_t normal_feature1,
  // Stage params
  int sample_index,
  uint denoise_available,
  uint2 pixel_dims);

KERNEL void kernel_raytrace(SceneParams params,
                            uint time,
                            image2d_write_t temp_color1,
                            uint2 pixel_dims,
                            GLOBAL TriangleData* triangles,
                            GLOBAL TriangleMetaData* tri_meta,
                            GLOBAL FlatBVHNode* bvh,
                            GLOBAL AreaLightData* lights,
                            uint num_lights,
                            image2d_array_read_t materials,
                            image2d_read_t sky,
                            uint denoise_available,
                            image2d_write_t albedo_feature1,
                            image2d_write_t normal_feature1);

KERNEL void kernel_accumulate(int sample_index,
                              uint denoise_available,
                              image2d_read_t temp_color1,
                              image2d_read_t albedo_feature1,
                              image2d_read_t normal_feature1,
                              image2d_read_t prev_color,
                              image2d_read_t prev_albedo_feature,
                              image2d_read_t prev_normal_feature,
                              image2d_write_t temp_color2,
                              image2d_write_t albedo_feature2,
                              image2d_write_t normal_feature2,
                              uint2 pixel_dims);

KERNEL void kernel_post_process(SceneParams params,
                                image2d_read_t temp_color2,
                                image2d_write_t pixels,
                                uint2 pixel_dims);

}

#endif