#include "kernels/anti_aliasing.hpp"
#include "kernels/backend/atomic.hpp"
#include "kernels/raytrace.hpp"
#include "kernels/transforms.hpp"

namespace nova {

KERNEL void kernel_raytrace(SceneParams params,
                            image2d_write_t temp_pixels1,
                            image2d_write_t temp_pixels2,
                            uint2 pixel_dims,
                            GLOBAL TriangleData* triangles,
                            GLOBAL TriangleMetaData* tri_meta,
                            GLOBAL FlatBVHNode* bvh,
                            image2d_array_read_t materials,
                            image2d_read_t sky) {
  int2 packed_pixel_coords = { static_cast<int>(get_global_id(0)),
                               static_cast<int>(get_global_id(1)) };
  if (packed_pixel_coords.x >= pixel_dims.x && packed_pixel_coords.y >= pixel_dims.y / 2) {
    return;
  }
  int2 pixel_coords = packed_pixel_coords;
  pixel_coords.y = 2 * pixel_coords.y + (pixel_coords.x & 1);

  float3 color = trace_ray(params, pixel_coords, triangles, tri_meta, bvh, materials, sky);

  write_image(temp_pixels1, packed_pixel_coords,
              make_vector<uchar4>(float3_to_uchar3(color), static_cast<unsigned char>(255)));
  write_image(temp_pixels2, pixel_coords, make_vector<float4>(color, 1.0f));
}

KERNEL void kernel_interpolate(image2d_read_t temp_pixels1,
                               image2d_write_t temp_pixels2,
                               uint2 pixel_dims,
                               GLOBAL uint* rem_pixels_counter,
                               GLOBAL int2* rem_coords) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y / 2) {
    return;
  }
  pixel_coords.y = 2 * pixel_coords.y + 1 - (pixel_coords.x & 1);

  // Sample 4 neighbours
  constexpr int2 neighbor_offsets[] = { { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 } };
  uint3 neighbors[4];
  for (uint i = 0; i < 4; i++) {
    // Lookup from the packed uchar4 texture
    int2 packed_coords = pixel_coords + neighbor_offsets[i];
    packed_coords.y = (packed_coords.y - (packed_coords.x & 1)) / 2;

    neighbors[i] = make_vector<uint3>(xyz<uchar3>(read_image<uchar4>(temp_pixels1, packed_coords)));
  }

  // Check color differences in the neighbours
  uint3 color_max = max(neighbors[0], max(neighbors[1], max(neighbors[2], neighbors[3])));
  uint3 color_min = min(neighbors[0], min(neighbors[1], min(neighbors[2], neighbors[3])));
  float3 color_range = uint3_to_float3(color_max - color_min);

  // If difference is large, raytrace to find color
  if (length(color_range) > INTERP_THRESHOLD) {
    rem_coords[atomic_inc(rem_pixels_counter)] = pixel_coords;
  }
  // Otherwise, interpolate
  else {
    uint3 color = (neighbors[0] + neighbors[1] + neighbors[2] + neighbors[3]) / 4U;
    write_image(temp_pixels2, pixel_coords, make_vector<float4>(uint3_to_float3(color), 1.0f));
  }
}

KERNEL void kernel_fill_remaining(SceneParams params,
                                  image2d_write_t temp_pixels2,
                                  uint2 pixel_dims,
                                  GLOBAL TriangleData* triangles,
                                  GLOBAL TriangleMetaData* tri_meta,
                                  GLOBAL FlatBVHNode* bvh,
                                  image2d_array_read_t materials,
                                  image2d_read_t sky,
                                  GLOBAL uint* rem_pixels_counter,
                                  GLOBAL int2* rem_coords) {
  uint id = static_cast<int>(get_global_id(0));
  if (id >= *rem_pixels_counter) {
    return;
  }
  int2 pixel_coords = rem_coords[id];

  float3 color = trace_ray(params, pixel_coords, triangles, tri_meta, bvh, materials, sky);

  write_image(temp_pixels2, pixel_coords, make_vector<float4>(color, 1.0f));
}

KERNEL void kernel_post_process(SceneParams params,
                                image2d_read_t temp_pixels2,
                                image2d_write_t pixels,
                                uint2 pixel_dims) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  float2 inv_pixel_dims = 1.0f / make_vector<float2>(pixel_dims);
  float2 pixel_uv = (make_vector<float2>(pixel_coords) + 0.5f) * inv_pixel_dims;

  float3 color;
  if (params.anti_aliasing) {
    color = fxaa(temp_pixels2, inv_pixel_dims, pixel_uv);
  } else {
    color = xyz<float3>(read_image<float4>(temp_pixels2, pixel_uv));
  }

  write_image(pixels, pixel_coords,
              make_vector<uchar4>(float3_to_uchar3(color), static_cast<unsigned char>(255)));
}

}