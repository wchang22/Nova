#ifndef KERNEL_RAYTRACE_HPP
#define KERNEL_RAYTRACE_HPP

#include "kernel_types/bvh_node.hpp"
#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"
#include "kernels/backend/image.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/constants.hpp"
#include "kernels/intersection.hpp"
#include "kernels/material.hpp"
#include "kernels/matrix.hpp"
#include "kernels/random.hpp"
#include "kernels/transforms.hpp"
#include "kernels/types.hpp"

namespace nova {

DEVICE bool find_intersection(
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

      if (!intersects_aabb(ray, xyz<float3>(node.top_offset_left),
                           xyz<float3>(node.bottom_num_right))) {
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
        stack[--tri_ptr] = (offset & TRIANGLE_OFFSET_MASK) | (num << TRIANGLE_NUM_SHIFT);

        node_index = stack[node_ptr--];
      }
      // Make sure tri_ptr and node_ptr do not collide
    } while (node_index && tri_ptr > node_ptr + 2);

    while (tri_ptr < STACK_SIZE) {
      // Pop list of triangles from stack back
      uint packed_triangle_data = stack[tri_ptr++];

      uint offset = packed_triangle_data & TRIANGLE_OFFSET_MASK;
      uint num = packed_triangle_data >> TRIANGLE_NUM_SHIFT;

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

DEVICE float3 trace_ray(uint& rng_state,
                        const SceneParams& params,
                        const int2& pixel_coords,
                        TriangleData* triangles,
                        TriangleMetaData* tri_meta,
                        FlatBVHNode* bvh,
                        image2d_array_read_t materials,
                        image2d_read_t sky) {
  // Jitter ray to get free anti-aliasing
  float2 offset = make_vector<float2>(rand(rng_state), rand(rng_state));

  float2 alpha_beta = params.eye_coords.coord_scale *
                      (make_vector<float2>(pixel_coords) - params.eye_coords.coord_dims + offset);
  float3 ray_dir = normalize(alpha_beta.x * params.eye_coords.eye_coord_frame.x -
                             alpha_beta.y * params.eye_coords.eye_coord_frame.y -
                             params.eye_coords.eye_coord_frame.z);
  float3 ray_pos = params.eye_coords.eye_pos;

  float3 color = make_vector<float3>(0.0f);
  float3 weight = make_vector<float3>(1.0f);
  bool indirect = false;

  while (true) {
    Ray ray(ray_pos, ray_dir, RAY_EPSILON);

    Intersection intrs;

    // Cast primary/reflection ray
    if (!find_intersection(triangles, bvh, ray, intrs, false)) {
      // TODO: IBL instead of just skymap
      if (!indirect) {
        color = read_sky(sky, ray_dir);
      }
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
                                   params.shading_diffuse) * meta.kD;
    // float metallic = read_material(materials, meta, texture_coord, meta.metallic_index,
    //                                make_vector<float3>(params.shading_metallic)).x;
    // float roughness = read_material(materials, meta, texture_coord, meta.roughness_index,
    //                                 make_vector<float3>(params.shading_roughness)).x;
    // float ambient_occlusion =
    //   read_material(materials, meta, texture_coord, meta.ambient_occlusion_index,
    //                 make_vector<float3>(params.shading_ambient_occlusion)).x;
    // clang-format on

    float3 normal = compute_normal(materials, meta, texture_coord, intrs.barycentric);

    // // Sample area light source
    // const AreaLight& light = params.light;
    // offset = light.size * make_vector<float2>(rand(rng_state) * 2.0f - 1.0f,
    //                                           rand(rng_state) * 2.0f - 1.0f);
    // Mat3x3 light_basis = create_basis(normalize(light.normal));
    // float3 light_position =
    //   light.position + light_basis * make_vector<float3>(offset.x, 0.0f, offset.y);

    // // Calculate lighting params
    // float3 light_dir = normalize(light_position - intrs_point);
    // float3 view_dir = -ray.direction;
    // float3 half_dir = normalize(light_dir + view_dir);
    // float light_distance = distance(light_position, intrs_point);
    // float3 kS = specularity(view_dir, half_dir, diffuse, metallic) * meta.kS;

    // float3 brdf = shade(params, light_dir, view_dir, half_dir, light_distance, normal,
    //                            diffuse, kS, metallic, roughness);

    // // Only cast a shadow ray if it will produce a color change
    // if (any(isgreaterequal(local_illum, make_vector<float3>(COLOR_EPSILON)))) {
    //   // Cast a shadow ray to the light
    //   Ray shadow_ray(intrs_point, light_dir, RAY_EPSILON);
    //   Intersection light_intrs;
    //   // Ensure objects blocking light are not behind the light
    //   light_intrs.length = light_distance;

    //   // Shade the pixel if ray is not blocked
    //   if (!find_intersection(triangles, bvh, shadow_ray, light_intrs, true)) {
    //     intrs_color += local_illum;
    //   }
    // }
    float3 brdf = diffuse / M_PI_F;

    color += weight * meta.kE;
    weight *= brdf * M_PI_F;

    // Russian roulette
    if (indirect) {
      float q = 1.0f - min(max(weight.x, max(weight.y, weight.z)), 1.0f);
      if (rand(rng_state) <= q) {
        break;
      }
      weight *= 1.0f / (1.0f - q);
    }

    // Use cosine importance sampling to sample direction
    Mat3x3 ray_basis = transpose(create_basis(normal));

    float phi = 2.0f * M_PI_F * rand(rng_state);
    float theta = acos(sqrt(rand(rng_state)));
    float3 random_dir =
      ray_basis * make_vector<float3>(sin(theta) * cos(phi), cos(theta), -sin(theta) * sin(phi));

    ray_pos = intrs_point;
    ray_dir = random_dir;

    indirect = true;
  }

  return color;
}

}

#endif