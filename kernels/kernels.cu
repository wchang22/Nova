#include "kernels/anti_aliasing.hpp"
#include "kernels/backend/atomic.hpp"
#include "kernels/constants.hpp"
#include "kernels/material.hpp"
#include "kernels/random.hpp"
#include "kernels/raytrace.hpp"
#include "kernels/transforms.hpp"
#include "kernels/types.hpp"

namespace nova {

KERNEL void kernel_generate(
  // Stage outputs
  GLOBAL PackedRay* rays,
  GLOBAL Path* paths,
  // Stage params
  SceneParams params,
  int sample_index,
  uint time,
  uint2 pixel_dims) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  uint path_index = pixel_coords.y * pixel_dims.x + pixel_coords.x;

  uint rng_state = hash(path_index + hash(time));

  rays[path_index] =
    generate_primary_ray(rng_state, params, pixel_coords).to_packed_ray(path_index);
  paths[path_index] = { make_vector<float3>(1.0f), make_vector<float3>(0.0f),
                        make_vector<float3>(0.0f), make_vector<float3>(0.0f), true };
}

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
  image2d_read_t sky) {
  uint id = get_global_id(0);
  if (id >= num_rays) {
    return;
  }

  PackedRay p_ray = rays[id];
  Ray ray(p_ray);
  Path& path = paths[get_path_index(p_ray)];

  Intersection intersection = intersect_ray(ray, path, triangles, bvh, sky, denoise_available);

  if (intersection.tri_index != -1) {
    intersection.ray_index = id;
    intersections[atomic_inc(intersection_count)] = intersection.to_intersection_data();
  }
}

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
  image2d_array_read_t materials) {
  uint id = get_global_id(0);
  if (id >= num_intersections) {
    return;
  }

  Intersection intersection(intersections[id]);
  uint rng_state = hash(id + hash(time));

  PackedRay p_ray = rays[intersection.ray_index];
  Ray ray(p_ray);
  uint path_index = get_path_index(p_ray);
  Path& path = paths[path_index];

  Ray extension_ray;
  if (shade_and_generate_extension_ray(rng_state, intersection, path, ray, params, triangles,
                                       tri_meta, bvh, lights, num_lights, materials,
                                       denoise_available, extension_ray)) {
    extended_rays[atomic_inc(ray_count)] = extension_ray.to_packed_ray(path_index);
  }
}

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
  uint2 pixel_dims) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  uint path_index = pixel_coords.y * pixel_dims.x + pixel_coords.x;
  const Path& path = paths[path_index];

  write_image(temp_color1, pixel_coords, make_vector<float4>(path.color, 1.0f));

  if (denoise_available) {
    write_image(albedo_feature1, pixel_coords, make_vector<float4>(path.albedo, 1.0f));
    write_image(normal_feature1, pixel_coords, make_vector<float4>(path.normal, 1.0f));
  }
}

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
                            image2d_write_t normal_feature1) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }
  uint rng_state = hash(pixel_coords.y * pixel_dims.x + pixel_coords.x + hash(time));

  float3 albedo;
  float3 normal;
  float3 color = trace_ray(rng_state, params, pixel_coords, triangles, tri_meta, bvh, lights,
                           num_lights, materials, sky, denoise_available, albedo, normal);

  write_image(temp_color1, pixel_coords, make_vector<float4>(color, 1.0f));

  if (denoise_available) {
    write_image(albedo_feature1, pixel_coords, make_vector<float4>(albedo, 1.0f));
    write_image(normal_feature1, pixel_coords, make_vector<float4>(normal, 1.0f));
  }
}

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
                              uint2 pixel_dims) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  float2 pixel_uv = coords_to_uv(pixel_coords, pixel_dims);
  float3 color = xyz<float3>(read_image<float4>(temp_color1, pixel_uv));
  float3 albedo = denoise_available ? xyz<float3>(read_image<float4>(albedo_feature1, pixel_uv))
                                    : make_vector<float3>(0.0f);
  float3 normal = denoise_available ? xyz<float3>(read_image<float4>(normal_feature1, pixel_uv))
                                    : make_vector<float3>(0.0f);

  float3 accumulated_color = color;
  float3 accumulated_albedo = albedo;
  float3 accumulated_normal = normal;

  if (sample_index != 0) {
    accumulated_color = xyz<float3>(read_image<float4>(prev_color, pixel_uv));
    accumulated_color =
      (accumulated_color * sample_index + color) / static_cast<float>(sample_index + 1);

    if (denoise_available) {
      accumulated_albedo = xyz<float3>(read_image<float4>(prev_albedo_feature, pixel_uv));
      accumulated_albedo =
        (accumulated_albedo * sample_index + albedo) / static_cast<float>(sample_index + 1);

      accumulated_normal = xyz<float3>(read_image<float4>(prev_normal_feature, pixel_uv));
      accumulated_normal = normalize((accumulated_normal * sample_index + normal) /
                                     static_cast<float>(sample_index + 1));
    }
  }

  write_image(temp_color2, pixel_coords, make_vector<float4>(accumulated_color, 1.0f));

  if (denoise_available) {
    write_image(albedo_feature2, pixel_coords, make_vector<float4>(accumulated_albedo, 1.0f));
    write_image(normal_feature2, pixel_coords, make_vector<float4>(accumulated_normal, 1.0f));
  }
}

KERNEL void kernel_post_process(SceneParams params,
                                image2d_read_t temp_color2,
                                image2d_write_t pixels,
                                uint2 pixel_dims) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  float2 pixel_uv = coords_to_uv(pixel_coords, pixel_dims);

  float3 color;
  if (params.anti_aliasing) {
    color = fxaa(temp_color2, 1.0f / make_vector<float2>(pixel_dims), pixel_uv);
  } else {
    color = xyz<float3>(read_image<float4>(temp_color2, pixel_uv));
  }

  color = gamma_correct(tone_map(color, params.exposure));

  write_image(pixels, pixel_coords,
              make_vector<uchar4>(float3_to_uchar3(color), static_cast<unsigned char>(255)));
}

}
