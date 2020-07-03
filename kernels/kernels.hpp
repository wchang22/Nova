#ifndef KERNEL_KERNELS_HPP
#define KERNEL_KERNELS_HPP

#include "kernel_types/bvh_node.hpp"
#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"
#include "kernels/backend/image.hpp"
#include "kernels/backend/kernel.hpp"

namespace nova {

KERNEL void kernel_raytrace(SceneParams params,
                            image2d_write_t temp_pixels1,
                            image2d_write_t temp_pixels2,
                            uint2 pixel_dims,
                            GLOBAL TriangleData* triangles,
                            GLOBAL TriangleMetaData* tri_meta,
                            GLOBAL FlatBVHNode* bvh,
                            image2d_array_read_t materials,
                            image2d_read_t sky);

KERNEL void kernel_interpolate(image2d_read_t temp_pixels1,
                               image2d_write_t temp_pixels2,
                               uint2 pixel_dims,
                               GLOBAL uint* rem_pixels_counter,
                               GLOBAL int2* rem_coords);

KERNEL void kernel_fill_remaining(SceneParams params,
                                  image2d_write_t temp_pixels2,
                                  uint2 pixel_dims,
                                  GLOBAL TriangleData* triangles,
                                  GLOBAL TriangleMetaData* tri_meta,
                                  GLOBAL FlatBVHNode* bvh,
                                  image2d_array_read_t materials,
                                  image2d_read_t sky,
                                  GLOBAL uint* rem_pixels_counter,
                                  GLOBAL int2* rem_coords);

KERNEL void kernel_post_process(SceneParams params,
                                image2d_read_t temp_pixels2,
                                image2d_write_t pixels,
                                uint2 pixel_dims);

}

#endif