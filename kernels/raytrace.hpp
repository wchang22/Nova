#ifndef KERNEL_RAYTRACE_HPP
#define KERNEL_RAYTRACE_HPP

#include "kernel_types/area_light.hpp"
#include "kernel_types/bvh_node.hpp"
#include "kernel_types/scene_params.hpp"
#include "kernel_types/triangle.hpp"
#include "kernels/backend/assertion.hpp"
#include "kernels/backend/image.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/brdf.hpp"
#include "kernels/constants.hpp"
#include "kernels/intersection.hpp"
#include "kernels/light.hpp"
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

      assert(node_ptr < STACK_SIZE);

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
                        AreaLightData* lights,
                        uint num_lights,
                        image2d_array_read_t materials,
                        image2d_read_t sky) {
  // Jitter ray to get free anti-aliasing
  float2 offset = make_vector<float2>(rand(rng_state), rand(rng_state));

  float2 alpha_beta = params.eye_coords.coord_scale *
                      (make_vector<float2>(pixel_coords) - params.eye_coords.coord_dims + offset);
  float3 ray_dir = normalize(transpose(params.eye_coords.eye_coord_frame) *
                             make_vector<float3>(alpha_beta.x, -alpha_beta.y, -1.0f));
  float3 ray_pos = params.eye_coords.eye_pos;

  float3 color = make_vector<float3>(0.0f);
  float3 weight = make_vector<float3>(1.0f);
  bool direct = true;

  while (true) {
    Ray ray(ray_pos, ray_dir, RAY_EPSILON);

    Intersection intrs;

    // Cast primary/reflection ray
    if (!find_intersection(triangles, bvh, ray, intrs, false)) {
      // TODO: IBL instead of just skymap
      if (direct) {
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
    float metallic = read_material(materials, meta, texture_coord, meta.metallic_index,
                                   make_vector<float3>(params.shading_metallic)).x;
    float roughness = read_material(materials, meta, texture_coord, meta.roughness_index,
                                    make_vector<float3>(params.shading_roughness)).x;
    // clang-format on

    float3 normal = compute_normal(materials, meta, texture_coord, intrs.barycentric);
    float3 out_dir = -ray_dir;

    float3 intrs_color = make_vector<float3>(0.0f);

    const auto estimate_direct_lighting = [&]() {
      // If there are no lights, or if the only light is the light we've intersected, don't
      // add any light contribution
      if (num_lights == 0 || (num_lights == 1 && meta.light_index != -1)) {
        return make_vector<float3>(0.0f);
      }

      // Randomly sample a single light
      uint random_light_index;
      do {
        random_light_index = min(static_cast<uint>(rand(rng_state) * num_lights), num_lights - 1);
      } while (meta.light_index == random_light_index);

      AreaLightData& light = lights[random_light_index];

      // Sample area light source
      float3 light_position = sample(light, rng_state);

      // Calculate lighting params
      float3 light_dir = normalize(light_position - intrs_point);
      float light_distance = distance(light_position, intrs_point);
      float light_area = light.dims.x * light.dims.y;

      // Cast a shadow ray to the light, ensuring objects blocking light are not behind the light
      Ray shadow_ray(intrs_point, light_dir, RAY_EPSILON);
      Intersection light_intrs(light_distance - RAY_EPSILON);

      // Add light contribution if ray is not blocked
      if (!find_intersection(triangles, bvh, shadow_ray, light_intrs, true)) {
        CookTorranceLightBRDF ct_light_brdf(light_dir, out_dir, normal, diffuse, metallic,
                                            roughness);
        float3 light_brdf = ct_light_brdf.eval();
        float light_pdf =
          ct_light_brdf.light_pdf(normalize(light.normal), light_distance, light_area);

        // Divide by (1 / num lights) to account for sampling a single light
        return num_lights * light.intensity * light_brdf / light_pdf *
               max(dot(normal, light_dir), 0.0f);
      }

      return make_vector<float3>(0.0f);
    };

    intrs_color += estimate_direct_lighting();

    // Only add emissive on first bounce, or if not a light
    intrs_color += (direct || meta.light_index == -1) ? meta.kE : make_vector<float3>(0.0f);

    // Sample material brdf for next direction
    CookTorranceBRDF ct_brdf(out_dir, normal, diffuse, metallic, roughness);

    float3 in_dir = ct_brdf.sample(rng_state);
    float3 brdf = ct_brdf.eval();
    float3 pdf = ct_brdf.pdf();

    color += weight * intrs_color;
    weight *= brdf / pdf * max(dot(normal, in_dir), 0.0f);

    assert(all(isfinite(in_dir)));
    assert(all(isfinite(color)));
    assert(all(isfinite(weight)));

    // Russian roulette
    if (!direct) {
      float p = min(max(weight.x, max(weight.y, weight.z)), 1.0f);
      if (rand(rng_state) > p) {
        break;
      }
      weight *= 1.0f / p;
    }

    ray_pos = intrs_point;
    ray_dir = in_dir;

    direct = false;
  }

  return color;
}

}

#endif