#include "backend/cuda/types/error.hpp"
#include "constants.hpp"
#include "intersection.hpp"
#include "raytrace.hpp"
#include "texture.hpp"
#include "transforms.hpp"
#include "types.hpp"
#include "vector_math.h"

__device__ bool find_intersection(
  TriangleData* triangles, FlatBVHNode* bvh, const Ray& ray, Intersection& min_intrs, bool fast) {
  /*
   * We maintain a double ended stack for space efficiency.
   * BVHNodes are pushed from the front to the back of the stack and
   * triangle offsets and nums are pushed from the back to the front of the stack.
   * This allows work items to find more than one leaf node before searching for
   * triangles and reduces branch divergence.
   */
  uint stack[STACK_SIZE];
  int node_ptr = -1;
  int tri_ptr = STACK_SIZE;

  // Set first value of stack to 0. We stop traversing when we pop this value.
  stack[++node_ptr] = 0;

  int node_index = 0;
  do {
    do {
      FlatBVHNode node = bvh[node_index];

      if (!intersects_aabb(ray, make_float3(node.top_offset_left),
                           make_float3(node.bottom_num_right))) {
        node_index = stack[node_ptr--];
        continue;
      }

      // Inner node, no triangles
      if (node.bottom_num_right.w >= 0) {
        // Traverse left and right children
        node_index = node.top_offset_left.w;
        stack[++node_ptr] = node.bottom_num_right.w;
      }
      // Leaf node, no children
      else {
        uint offset = node.top_offset_left.w;
        uint num = -node.bottom_num_right.w;

        // Pack offset and num into a single uint to save memory and push to stack back
        stack[--tri_ptr] =
          (offset & constants.triangle_offset_mask) | (num << constants.triangle_num_shift);

        node_index = stack[node_ptr--];
      }
      // Make sure tri_ptr and node_ptr do not collide
    } while (node_index && tri_ptr > node_ptr + 2);

    while (tri_ptr < STACK_SIZE) {
      // Pop list of triangles from stack back
      uint packed_triangle_data = stack[tri_ptr++];

      uint offset = packed_triangle_data & constants.triangle_offset_mask;
      uint num = packed_triangle_data >> constants.triangle_num_shift;

      // If intersected, compute intersection for all triangles in the node
      for (uint i = offset; i < offset + num; i++) {
        if (intersects_triangle(ray, min_intrs, i, triangles[i]) && fast) {
          return true;
        }
      }
    }
  } while (node_index);

  return min_intrs.tri_index != -1;
}

__device__ float3 trace_ray(uint2 pixel_coords,
                            TriangleData* triangles,
                            TriangleMetaData* tri_meta,
                            FlatBVHNode* bvh,
                            cudaTextureObject_t materials) {
  float2 alpha_beta = params.eye_coords.coord_scale *
                      (make_float2(pixel_coords) - params.eye_coords.coord_dims + 0.5f);
  float3 ray_dir = normalize(alpha_beta.x * params.eye_coords.eye_coord_frame.x -
                             alpha_beta.y * params.eye_coords.eye_coord_frame.y -
                             params.eye_coords.eye_coord_frame.z);
  float3 ray_pos = params.eye_coords.eye_pos;

  float3 color = make_float3(0.0f);
  float3 reflectance = make_float3(1.0f);

  for (int depth = 0; depth < params.ray_bounces; depth++) {
    Ray ray = create_ray(ray_pos, ray_dir, RAY_EPSILON);

    Intersection intrs = no_intersection();

    // Cast primary/reflection ray
    if (!find_intersection(triangles, bvh, ray, intrs, false)) {
      break;
    }

    TriangleMetaData meta = tri_meta[intrs.tri_index];

    // Calculate intersection point
    float3 intrs_point = ray.origin + ray.direction * intrs.length;

    // Interpolate texture coords from vertex data
    float2 texture_coord = triangle_interpolate(intrs.barycentric, meta.texture_coord1,
                                                meta.texture_coord2, meta.texture_coord3);

    // Look up materials
    // clang-format off
    float3 diffuse = read_material(materials, meta, texture_coord, meta.diffuse_index,
                                   params.shading_diffuse);
    float metallic = read_material(materials, meta, texture_coord, meta.metallic_index,
                                   make_float3(params.shading_metallic)).x;
    float roughness = read_material(materials, meta, texture_coord, meta.roughness_index,
                                    make_float3(params.shading_roughness)).x;
    float ambient_occlusion =
      read_material(materials, meta, texture_coord, meta.ambient_occlusion_index,
                    make_float3(params.shading_ambient_occlusion)).x;
    // clang-format on

    float3 normal = compute_normal(materials, meta, texture_coord, intrs.barycentric);

    // Add ambient color even if pixel is in shadow
    float3 intrs_color = diffuse * ambient_occlusion * 0.03f;

    // Calculate lighting params
    float3 light_dir = normalize(params.light_position - intrs_point);
    float3 view_dir = -ray.direction;
    float3 half_dir = normalize(light_dir + view_dir);
    float light_distance = distance(params.light_position, intrs_point);
    float3 kS = specularity(view_dir, half_dir, diffuse, metallic);

    // Cast a shadow ray to the light
    Ray shadow_ray = create_ray(intrs_point, light_dir, RAY_EPSILON);
    Intersection light_intrs = no_intersection();
    // Ensure objects blocking light are not behind the light
    light_intrs.length = light_distance;

    // Shade the pixel if ray is not blocked
    if (!find_intersection(triangles, bvh, shadow_ray, light_intrs, true)) {
      intrs_color += shade(light_dir, view_dir, half_dir, light_distance, normal, diffuse, kS,
                           metallic, roughness);
    }

    /*
     * Normally, color is calculated recursively:
     * (intrs_color + specular * (intrs_color of reflected ray))
     * So we use an additional "reflectance" value to unroll the recursion
     */
    color += reflectance * intrs_color;
    reflectance *= kS;

    // Stop if reflectance is too low to produce a color change
    if (all(isless(reflectance, COLOR_EPSILON))) {
      break;
    }

    // Reflect ray off of intersection point
    ray_pos = intrs_point;
    ray_dir = reflect(ray_dir, normal);
  }

  return clamp(color, 0.0f, 1.0f);
}

__global__ void raytrace(uchar4* pixels,
                         uint2 pixel_dims,
                         TriangleData* triangles,
                         TriangleMetaData* tri_meta,
                         FlatBVHNode* bvh,
                         cudaTextureObject_t materials) {
  uint2 pixel_coords = { blockDim.x * blockIdx.x + threadIdx.x,
                         blockDim.y * blockIdx.y + threadIdx.y };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y / 2) {
    return;
  }
  pixel_coords.y = 2 * pixel_coords.y + (pixel_coords.x & 1);

  float3 color = trace_ray(pixel_coords, triangles, tri_meta, bvh, materials);

  int pixel_index = linear_index(make_int2(pixel_coords), pixel_dims.x);
  pixels[pixel_index] = make_uchar4(make_float4(color, 1.0f) * 255.0f);
}

__global__ void interpolate(uchar4* pixels,
                            uint2 pixel_dims,
                            TriangleData* triangles,
                            TriangleMetaData* tri_meta,
                            FlatBVHNode* bvh,
                            cudaTextureObject_t materials,
                            uint* rem_pixels_counter,
                            uint2* rem_coords) {
  uint2 pixel_coords = { blockDim.x * blockIdx.x + threadIdx.x,
                         blockDim.y * blockIdx.y + threadIdx.y };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y / 2) {
    return;
  }
  pixel_coords.y = 2 * pixel_coords.y + 1 - (pixel_coords.x & 1);

  // Sample 4 neighbours
  int2 neighbor_offsets[] = { { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 } };
  uint3 neighbors[4];
  for (uint i = 0; i < 4; i++) {
    int index = linear_index(
      clamp(make_int2(pixel_coords) + neighbor_offsets[i], make_int2(0), make_int2(pixel_dims) - 1),
      pixel_dims.x);
    neighbors[i] = make_uint3(make_uchar3(pixels[index]));
  }

  // Check color differences in the neighbours
  uint3 color_max = max(neighbors[0], max(neighbors[1], max(neighbors[2], neighbors[3])));
  uint3 color_min = min(neighbors[0], min(neighbors[1], min(neighbors[2], neighbors[3])));
  float3 color_range = make_float3(color_max - color_min) / 255.0f;

  // If difference is large, raytrace to find color
  if (length(color_range) > INTERP_THRESHOLD) {
    rem_coords[atomicAdd(rem_pixels_counter, 1)] = pixel_coords;
  }
  // Otherwise, interpolate
  else {
    int pixel_index = linear_index(make_int2(pixel_coords), pixel_dims.x);
    uint3 color = (neighbors[0] + neighbors[1] + neighbors[2] + neighbors[3]) / 4U;
    pixels[pixel_index] = make_uchar4(make_uint4(color, 255));
  }
}

__global__ void fill_remaining(uchar4* pixels,
                               uint2 pixel_dims,
                               TriangleData* triangles,
                               TriangleMetaData* tri_meta,
                               FlatBVHNode* bvh,
                               cudaTextureObject_t materials,
                               uint* rem_pixels_counter,
                               uint2* rem_coords) {
  uint id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= *rem_pixels_counter) {
    return;
  }
  uint2 pixel_coords = rem_coords[id];

  float3 color = trace_ray(pixel_coords, triangles, tri_meta, bvh, materials);

  int pixel_index = linear_index(make_int2(pixel_coords), pixel_dims.x);
  pixels[pixel_index] = make_uchar4(make_float4(color, 1.0f) * 255.0f);
}

void kernel_raytrace(uint2 global_dims,
                     uint2 local_dims,
                     const KernelConstants& kernel_constants,
                     const SceneParams& scene_params,
                     uchar4* pixels,
                     uint2 pixel_dims,
                     TriangleData* triangles,
                     TriangleMetaData* tri_meta,
                     FlatBVHNode* bvh,
                     cudaTextureObject_t materials) {
  dim3 block_size { local_dims.x, local_dims.y, 1 };
  dim3 num_blocks { global_dims.x / block_size.x, global_dims.y / block_size.y, 1 };
  CUDA_CHECK_AND_THROW(cudaMemcpyToSymbol(constants, &kernel_constants, sizeof(KernelConstants), 0,
                                          cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_THROW(
    cudaMemcpyToSymbol(params, &scene_params, sizeof(SceneParams), 0, cudaMemcpyHostToDevice));
  raytrace<<<num_blocks, block_size>>>(pixels, pixel_dims, triangles, tri_meta, bvh, materials);
}

void kernel_interpolate(uint2 global_dims,
                        uint2 local_dims,
                        const KernelConstants& kernel_constants,
                        uchar4* pixels,
                        uint2 pixel_dims,
                        TriangleData* triangles,
                        TriangleMetaData* tri_meta,
                        FlatBVHNode* bvh,
                        cudaTextureObject_t materials,
                        uint* rem_pixels_counter,
                        uint2* rem_coords) {
  dim3 block_size { local_dims.x, local_dims.y, 1 };
  dim3 num_blocks { global_dims.x / block_size.x, global_dims.y / block_size.y, 1 };
  CUDA_CHECK_AND_THROW(cudaMemcpyToSymbol(constants, &kernel_constants, sizeof(KernelConstants), 0,
                                          cudaMemcpyHostToDevice));
  interpolate<<<num_blocks, block_size>>>(pixels, pixel_dims, triangles, tri_meta, bvh, materials,
                                          rem_pixels_counter, rem_coords);
}

void kernel_fill_remaining(uint2 global_dims,
                           uint2 local_dims,
                           const KernelConstants& kernel_constants,
                           const SceneParams& scene_params,
                           uchar4* pixels,
                           uint2 pixel_dims,
                           TriangleData* triangles,
                           TriangleMetaData* tri_meta,
                           FlatBVHNode* bvh,
                           cudaTextureObject_t materials,
                           uint* rem_pixels_counter,
                           uint2* rem_coords) {
  dim3 block_size { local_dims.x, local_dims.y, 1 };
  dim3 num_blocks { global_dims.x / block_size.x, global_dims.y / block_size.y, 1 };
  CUDA_CHECK_AND_THROW(cudaMemcpyToSymbol(constants, &kernel_constants, sizeof(KernelConstants), 0,
                                          cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_THROW(
    cudaMemcpyToSymbol(params, &scene_params, sizeof(SceneParams), 0, cudaMemcpyHostToDevice));
  fill_remaining<<<num_blocks, block_size>>>(pixels, pixel_dims, triangles, tri_meta, bvh,
                                             materials, rem_pixels_counter, rem_coords);
}