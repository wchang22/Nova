#include "kernels/anti_aliasing.hpp"
#include "kernels/backend/atomic.hpp"
#include "kernels/random.hpp"
#include "kernels/raytrace.hpp"
#include "kernels/transforms.hpp"

namespace nova {

KERNEL void kernel_raytrace(SceneParams params,
                            uint time,
                            image2d_array_write_t temp_img1,
                            uint2 pixel_dims,
                            GLOBAL TriangleData* triangles,
                            GLOBAL TriangleMetaData* tri_meta,
                            GLOBAL FlatBVHNode* bvh,
                            GLOBAL AreaLightData* lights,
                            uint num_lights,
                            image2d_array_read_t materials,
                            image2d_read_t sky,
                            uint denoise_available) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }
  uint rng_state = hash(pixel_coords.y * pixel_dims.x + pixel_coords.x + hash(time));

  float3 albedo;
  float3 normal;
  float3 color = trace_ray(rng_state, params, pixel_coords, triangles, tri_meta, bvh, lights,
                           num_lights, materials, sky, denoise_available, albedo, normal);

  write_image(temp_img1, pixel_coords, 0, make_vector<float4>(color, 1.0f));

  if (denoise_available) {
    write_image(temp_img1, pixel_coords, 1, make_vector<float4>(albedo, 1.0f));
    write_image(temp_img1, pixel_coords, 2, make_vector<float4>(normal, 1.0f));
  }
}

KERNEL void kernel_accumulate(int sample_index,
                              uint denoise_available,
                              image2d_array_read_t temp_img1,
                              image2d_array_read_t prev_img,
                              image2d_array_write_t temp_img2,
                              uint2 pixel_dims) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  float2 pixel_uv = coords_to_uv(pixel_coords, pixel_dims);
  float3 color = xyz<float3>(read_image<float4>(temp_img1, pixel_uv, 0));
  float3 albedo = denoise_available ? xyz<float3>(read_image<float4>(temp_img1, pixel_uv, 1))
                                    : make_vector<float3>(0.0f);
  float3 normal = denoise_available ? xyz<float3>(read_image<float4>(temp_img1, pixel_uv, 2))
                                    : make_vector<float3>(0.0f);

  float3 accumulated_color = color;
  float3 accumulated_albedo = albedo;
  float3 accumulated_normal = normal;

  if (sample_index != 0) {
    accumulated_color = xyz<float3>(read_image<float4>(prev_img, pixel_uv, 0));
    accumulated_color =
      (accumulated_color * sample_index + color) / static_cast<float>(sample_index + 1);

    if (denoise_available) {
      accumulated_albedo = xyz<float3>(read_image<float4>(prev_img, pixel_uv, 1));
      accumulated_albedo =
        (accumulated_albedo * sample_index + albedo) / static_cast<float>(sample_index + 1);

      accumulated_normal = xyz<float3>(read_image<float4>(prev_img, pixel_uv, 2));
      accumulated_normal = normalize((accumulated_normal * sample_index + normal) /
                                     static_cast<float>(sample_index + 1));
    }
  }

  write_image(temp_img2, pixel_coords, 0, make_vector<float4>(accumulated_color, 1.0f));

  if (denoise_available) {
    write_image(temp_img2, pixel_coords, 1, make_vector<float4>(accumulated_albedo, 1.0f));
    write_image(temp_img2, pixel_coords, 2, make_vector<float4>(accumulated_normal, 1.0f));
  }
}

KERNEL void kernel_post_process(SceneParams params,
  image2d_array_read_t temp_img2,
                                image2d_write_t pixels,
                                uint2 pixel_dims) {
  int2 pixel_coords = { static_cast<int>(get_global_id(0)), static_cast<int>(get_global_id(1)) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  float2 pixel_uv = coords_to_uv(pixel_coords, pixel_dims);

  float3 color;
  if (params.anti_aliasing) {
    color = fxaa(temp_img2, 1.0f / make_vector<float2>(pixel_dims), pixel_uv);
  } else {
    color = xyz<float3>(read_image<float4>(temp_img2, pixel_uv, 0));
  }

  color = gamma_correct(tone_map(color, params.exposure));

  write_image(pixels, pixel_coords,
              make_vector<uchar4>(float3_to_uchar3(color), static_cast<unsigned char>(255)));
}

}
