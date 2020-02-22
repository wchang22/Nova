#include "raytrace.h"
#include "constants.h"
#include "vector_math.h"
#include "intersection.h"
#include "texture.h"
#include "transforms.h"
#include "types.h"
#include "backend/cuda/types/types.h"

__device__
bool trace(TriangleData* triangles, FlatBVHNode* bvh, Ray ray, Intersection* min_intrs, bool fast) {
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
        uint left = node.top_offset_left.w;
        uint right = node.bottom_num_right.w;

        if (!left && !right) {
          node_index = stack[node_ptr--];
        } else {
          node_index = left ? left : right;
          if (left && right) {
            stack[++node_ptr] = right;
          }
        }
      }
      // Leaf node, no children
      else {
        uint offset = node.top_offset_left.w;
        uint num = -node.bottom_num_right.w;

        // Pack offset and num into a single uint to save memory
        uint packed_triangle_data =
          (offset & constants.triangle_offset_mask) | (num << constants.triangle_num_shift);

        // Push list of triangles to stack back
        stack[--tri_ptr] = packed_triangle_data;

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

  return min_intrs->tri_index != -1;
}

__global__
void raytrace(cudaSurfaceObject_t image_out,
              EyeCoords ec,
              TriangleData* triangles,
              TriangleMetaData* tri_meta,
              FlatBVHNode* bvh,
              cudaTextureObject_t materials) {
  uint2 pixel_coords = { blockDim.x * blockIdx.x + threadIdx.x,
                         blockDim.y * blockIdx.y + threadIdx.y };
  
  float2 alpha_beta = ec.coord_scale * (make_float2(pixel_coords) - ec.coord_dims + 0.5f);
  float3 ray_dir = normalize(alpha_beta.x * ec.eye_coord_frame.x -
                             alpha_beta.y * ec.eye_coord_frame.y -
                                            ec.eye_coord_frame.z);
  float3 ray_pos = ec.eye_pos;

  float3 color = make_float3(0.0f);
  float3 reflectance = make_float3(1.0f);

  for (int depth = 0; depth < constants.ray_recursion_depth; depth++) {
    Ray ray = create_ray(ray_pos, ray_dir, RAY_EPSILON);

    Intersection intrs = NO_INTERSECTION;

    // Cast primary/reflection ray
    if (!trace(triangles, bvh, ray, &intrs, false)) {
      break;
    }

    TriangleMetaData meta = tri_meta[intrs.tri_index];

    // Calculate intersection point
    float3 intrs_point = ray.origin + ray.direction * intrs.length;

    // Interpolate texture coords from vertex data
    float2 texture_coord = triangle_interpolate(
      intrs.barycentric, meta.texture_coord1, meta.texture_coord2, meta.texture_coord3
    );

    // Look up materials
    float3 diffuse = read_material(materials, meta, texture_coord, meta.diffuse_index,
                                   constants.default_diffuse);
    float metallic = read_material(materials, meta, texture_coord, meta.metallic_index,
                                   make_float3(constants.default_metallic)).x;
    float roughness = read_material(materials, meta, texture_coord, meta.roughness_index,       
                                    make_float3(constants.default_roughness)).x;
    float ambient_occlusion = read_material(materials, meta, texture_coord,   
                                            meta.ambient_occlusion_index,
                                            make_float3(constants.default_ambient_occlusion)).x;

    float3 normal = compute_normal(materials, meta, texture_coord, intrs.barycentric);

    // Add ambient color even if pixel is in shadow
    float3 intrs_color = diffuse * ambient_occlusion * 0.03f;

    // Calculate lighting params
    float3 light_dir = normalize(constants.light_position - intrs_point);
    float3 view_dir = normalize(ec.eye_pos - intrs_point);
    float3 half_dir = normalize(light_dir + view_dir);
    float light_distance = distance(constants.light_position, intrs_point);
    float3 kS = specularity(view_dir, half_dir, diffuse, metallic);

    // Cast a shadow ray to the light
    Ray shadow_ray = create_ray(intrs_point, light_dir, RAY_EPSILON);
    Intersection light_intrs = NO_INTERSECTION;
    // Ensure objects blocking light are not behind the light
    light_intrs.length = light_distance;

    // Shade the pixel if ray is not blocked
    if (!trace(triangles, bvh, shadow_ray, &light_intrs, true)) {
      intrs_color += shade(light_dir, view_dir, half_dir, light_distance,
                          normal, diffuse, kS, metallic, roughness);
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

  uchar4 image_color = make_uchar4(make_float4(color, 1.0f) * 255.0f);
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
  CUDA_CHECK(cudaMemcpyToSymbol(constants, &kernel_constants,
                                sizeof(KernelConstants), 0, cudaMemcpyHostToDevice));
  raytrace<<<num_blocks, block_size>>>(image_out, ec, triangles, tri_meta, bvh, materials);
}