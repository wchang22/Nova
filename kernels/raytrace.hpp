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

struct Leaf {
  uint offset;
  uint num;

  DEVICE bool is_empty() const {
    return num == 0;
  }
};

DEVICE bool find_intersection(
  TriangleData* triangles, FlatBVHNode* bvh, const Ray& ray, Intersection& min_intrs, bool anyhit) {
  uint stack[STACK_SIZE];
  int node_ptr = -1;

  // Set first value of stack to 0. We stop traversing when we pop this value.
  stack[++node_ptr] = 0;

  int node_index = 0;
  do {
    Leaf leaf {};
    Leaf postponed_leaf {};

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

        // Left child is adjacent to parent
        node_index++;
        // Put right child on the stack
        stack[++node_ptr] = node.bottom_num_right.w;
      }
      // Leaf node, no children
      else {
        uint offset = node.top_offset_left.w;
        uint num = -node.bottom_num_right.w;
        assert(num != 0);

        node_index = stack[node_ptr--];

        if (postponed_leaf.is_empty()) {
          postponed_leaf.offset = offset;
          postponed_leaf.num = num;
        } else {
          leaf.offset = offset;
          leaf.num = num;
          break;
        }
      }

      assert(node_ptr < STACK_SIZE);
    } while (node_index);

    Leaf leaves[2] = { postponed_leaf, leaf };

    for (const auto& l : leaves) {
      if (l.is_empty()) {
        continue;
      }

      // If intersected, compute intersection for all triangles in the node
      for (uint i = l.offset; i < l.offset + l.num; i++) {
        if (intersects_triangle(ray, min_intrs, i, triangles[i]) && anyhit) {
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
                        image2d_read_t sky,
                        bool denoise_available,
                        float3& albedo_feature,
                        float3& normal_feature) {
  // Jitter ray to get free anti-aliasing
  float2 offset = make_vector<float2>(rand(rng_state), rand(rng_state));

  float2 alpha_beta = params.eye_coords.coord_scale *
                      (make_vector<float2>(pixel_coords) - params.eye_coords.coord_dims + offset);
  float3 ray_dir = normalize(transpose(params.eye_coords.eye_coord_frame) *
                             make_vector<float3>(alpha_beta.x, -alpha_beta.y, -1.0f));
  float3 ray_pos = params.eye_coords.eye_pos;

  float3 color = make_vector<float3>(0.0f);
  float3 throughput = make_vector<float3>(1.0f);

  bool direct = true;

  while (true) {
    Ray ray(ray_pos, ray_dir, RAY_EPSILON);

    Intersection intrs;

    // Cast primary/reflection ray
    if (!find_intersection(triangles, bvh, ray, intrs, false)) {
      // TODO: IBL instead of just skymap
      if (direct) {
        color = read_sky(sky, ray_dir);
        if (denoise_available) {
          albedo_feature = color / (color + 1.0f);
          normal_feature = make_vector<float3>(0.0f);
        }
      }
      break;
    }

    const TriangleMetaData& meta = tri_meta[intrs.tri_index];

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
                                   make_vector<float3>(meta.metallic == -1.0f ?
                                   params.shading_metallic : meta.metallic)).x;
    float roughness = read_material(materials, meta, texture_coord, meta.roughness_index,
                                    make_vector<float3>(meta.roughness == -1.0f ?
                                    params.shading_roughness : meta.roughness)).x;
    // clang-format on

    if (!params.path_tracing) {
      if (meta.light_index != -1) {
        color += lights[meta.light_index].intensity;
      } else {
        color += diffuse;
      }
      break;
    }

    float3 normal = compute_normal(materials, meta, texture_coord, intrs.barycentric);
    float3 out_dir = -ray_dir;

    if (direct && denoise_available) {
      albedo_feature = diffuse;
      normal_feature = normal;
    }

    // Only add light on first bounce to prevent double counting
    if (direct && meta.light_index != -1) {
      color += throughput * lights[meta.light_index].intensity;
    }

    // Estimate direct lighting
    float3 direct_color = make_vector<float3>(0.0f);

    // Only sample direct lighting if more than 1 light or if only one light and we didn't hit any
    if (num_lights > 1 || (num_lights == 1 && meta.light_index == -1)) {
      {
        // Randomly sample a single light
        uint random_light_index;
        do {
          random_light_index = min(static_cast<uint>(rand(rng_state) * num_lights), num_lights - 1);
        } while (meta.light_index == random_light_index);

        const AreaLightData& light = lights[random_light_index];
        float3 light_normal = compute_normal(light);
        float light_area = compute_area(light);

        // Sample area light source
        float3 light_position = sample(light, rng_state);

        // Calculate lighting params
        float3 light_dir = normalize(light_position - intrs_point);
        float light_distance = distance(light_position, intrs_point);

        // Ensuring objects blocking light are not behind the light
        Ray light_ray(intrs_point, light_dir, RAY_EPSILON);
        Intersection light_intrs(light_distance - RAY_EPSILON);

        // Multiple importance sample lights: Add light contribution if ray is not blocked
        if (!find_intersection(triangles, bvh, light_ray, light_intrs, true)) {
          CookTorranceLightBRDF ct_light_brdf(light_dir, out_dir, normal, diffuse, metallic,
                                              roughness);
          float3 light_brdf = ct_light_brdf.eval();
          float light_pdf = ct_light_brdf.light_pdf(light_normal, light_distance, light_area);
          float3 light_brdf_pdf = ct_light_brdf.pdf();
          float3 weight = power_heuristic(make_vector<float3>(light_pdf), light_brdf_pdf);

          // Divide by (1 / num lights) to account for sampling a single light
          direct_color += num_lights * weight * light.intensity * light_brdf / light_pdf *
                          max(dot(normal, light_dir), 0.0f);
        }
      }

      {
        // Multiple importance sample brdf for lighting
        CookTorranceLightBRDF ct_light_brdf({}, out_dir, normal, diffuse, metallic, roughness);
        float3 light_dir = ct_light_brdf.sample(rng_state);

        Ray light_ray(intrs_point, light_dir, RAY_EPSILON);
        Intersection light_intrs;

        // Add light contribution if intersected light
        if (find_intersection(triangles, bvh, light_ray, light_intrs, false)) {
          const TriangleMetaData& light_meta = tri_meta[light_intrs.tri_index];

          // Make sure we actually hit the light
          if (light_meta.light_index != -1 && light_meta.light_index != meta.light_index) {
            float3 light_position = light_ray.origin + light_ray.direction * light_intrs.length;
            float light_distance = light_intrs.length;

            const AreaLightData& light = lights[light_meta.light_index];
            float3 light_normal = compute_normal(light);
            float light_area = compute_area(light);

            float3 light_brdf = ct_light_brdf.eval();
            float light_pdf = ct_light_brdf.light_pdf(light_normal, light_distance, light_area);
            float3 light_brdf_pdf = ct_light_brdf.pdf();
            float3 weight = power_heuristic(light_brdf_pdf, make_vector<float3>(light_pdf));

            direct_color += weight * light.intensity * light_brdf / light_brdf_pdf *
                            max(dot(normal, light_dir), 0.0f);
          }
        }
      }
    }

    color += throughput * direct_color;

    // Sample material brdf for next direction
    CookTorranceBRDF ct_brdf(out_dir, normal, diffuse, metallic, roughness);

    float3 in_dir = ct_brdf.sample(rng_state);
    float3 brdf = ct_brdf.eval();
    float3 pdf = ct_brdf.pdf();

    throughput *= brdf / pdf * max(dot(normal, in_dir), 0.0f);

    assert(all(isfinite(in_dir)));
    assert(all(isfinite(color)));
    assert(all(isfinite(throughput)));

    // Russian roulette
    if (!direct) {
      float p = min(max(throughput.x, max(throughput.y, throughput.z)), 1.0f);
      if (rand(rng_state) > p) {
        break;
      }
      throughput *= 1.0f / p;
    }

    ray_pos = intrs_point;
    ray_dir = in_dir;

    direct = false;
  }

  return color;
}

}

#endif