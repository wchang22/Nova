#ifndef KERNEL_ANTI_ALIASING_HPP
#define KERNEL_ANTI_ALIASING_HPP

#include "kernels/backend/image.hpp"
#include "kernels/backend/kernel.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/new/constants.hpp"
#include "kernels/new/transforms.hpp"

namespace nova {

// Fast approximate anti-aliasing:
// http://blog.simonrodriguez.fr/articles/30-07-2016_implementing_fxaa.html
DEVICE inline float3 fxaa(image2d_read_t pixels, float2 inv_pixel_dims, float2 pixel_uv) {
  float3 center_color = xyz<float3>(read_image<float4>(pixels, pixel_uv));
  float center = rgb_to_luma(center_color);

  const float2 offsets[] = {
    make_vector<float2>(0.0f, -1.0f) * inv_pixel_dims,
    make_vector<float2>(-1.0f, 0.0f) * inv_pixel_dims,
    make_vector<float2>(1.0f, 0.0f) * inv_pixel_dims,
    make_vector<float2>(0.0f, 1.0f) * inv_pixel_dims,
    make_vector<float2>(-1.0f, -1.0f) * inv_pixel_dims,
    make_vector<float2>(1.0f, -1.0f) * inv_pixel_dims,
    make_vector<float2>(-1.0f, 1.0f) * inv_pixel_dims,
    make_vector<float2>(1.0f, 1.0f) * inv_pixel_dims,
  };

  // Sample 4 neighbours
  float down = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[0])));
  float left = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[1])));
  float right = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[2])));
  float up = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[3])));

  float luma_max = max(center, max(up, max(left, max(right, down))));
  float luma_min = min(center, min(up, min(left, min(right, down))));
  float luma_range = luma_max - luma_min;

  // Don't need AA if not an edge
  if (luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD_MAX)) {
    return center_color;
  }

  // Sample other 4 neighbours
  float down_left = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[4])));
  float down_right = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[5])));
  float up_left = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[6])));
  float up_right = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, pixel_uv, offsets[7])));

  float down_up = down + up;
  float left_right = left + right;

  float left_corners = down_left + up_left;
  float down_corners = down_left + down_right;
  float right_corners = down_right + up_right;
  float up_corners = up_left + up_right;

  // Detect edges
  float edge_h = fabs(left_corners - 2.0f * left) + 2 * fabs(down_up - 2.0f * center) +
                 fabs(right_corners - 2.0f * right);
  float edge_v = fabs(up_corners - 2.0f * up) + 2 * fabs(left_right - 2.0f * center) +
                 fabs(down_corners - 2.0f * down);

  bool is_horizontal = edge_h >= edge_v;

  // Calculate gradient opposite edge direction
  float2 luma = is_horizontal ? make_vector<float2>(down, up) : make_vector<float2>(left, right);
  float2 gradient = luma - center;
  float gradient_scaled = 0.25f * max(fabs(gradient.x), fabs(gradient.y));

  // Compute average luma on the edge
  float step_length = is_horizontal ? inv_pixel_dims.y : inv_pixel_dims.x;
  float luma_local_avg;

  if (fabs(gradient.x) >= fabs(gradient.y)) {
    step_length = -step_length;
    luma_local_avg = 0.5f * (luma.x + center);
  } else {
    luma_local_avg = 0.5f * (luma.y + center);
  }

  float2 uv = pixel_uv;
  if (is_horizontal) {
    uv.y += 0.5f * step_length;
  } else {
    uv.x += 0.5f * step_length;
  }

  // Explore luma along edge until we reach the edges
  float2 offset = is_horizontal ? make_vector<float2>(inv_pixel_dims.x, 0.0f)
                                : make_vector<float2>(0.0f, inv_pixel_dims.y);
  float2 uv1 = uv - offset;
  float2 uv2 = uv + offset;
  bool reached_end1 = false;
  bool reached_end2 = false;
  float luma_end1;
  float luma_end2;

  const float step_mod[EDGE_SEARCH_ITERATIONS] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.5f,
                                                   2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 8.0f };

  for (uint i = 0; i < EDGE_SEARCH_ITERATIONS; i++) {
    if (!reached_end1) {
      luma_end1 = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, uv1))) - luma_local_avg;
      reached_end1 = fabs(luma_end1) >= gradient_scaled;
      uv1 -= step_mod[i] * offset;
    }
    if (!reached_end2) {
      luma_end2 = rgb_to_luma(xyz<float3>(read_image<float4>(pixels, uv2))) - luma_local_avg;
      reached_end2 = fabs(luma_end2) >= gradient_scaled;
      uv2 += step_mod[i] * offset;
    }
    if (reached_end1 && reached_end2) {
      break;
    }
  }

  // Calculate distance to end
  float dist1 = is_horizontal ? (pixel_uv.x - uv1.x) : (pixel_uv.y - uv1.y);
  float dist2 = is_horizontal ? (uv2.x - pixel_uv.x) : (uv2.y - pixel_uv.y);

  float dist_final = min(dist1, dist2);

  float edge_thickness = dist1 + dist2;
  float pixel_offset = -dist_final / edge_thickness + 0.5f;

  bool is_luma_center_smaller = center < luma_local_avg;
  bool correct_variation =
    ((dist1 < dist2 ? luma_end1 : luma_end2) < 0.0f) != is_luma_center_smaller;
  float final_offset = correct_variation ? pixel_offset : 0.0f;

  // Subpixel aliasing
  float luma_avg = (1.0f / 12.0f) * (2.0f * (down_up + left_right) + left_corners + right_corners);

  float subpixel_offset1 = clamp(fabs(luma_avg - center) / luma_range, 0.0f, 1.0f);
  float subpixel_offset2 = (-2.0f * subpixel_offset1 + 3.0f) * subpixel_offset1 * subpixel_offset1;
  float subpixel_offset_final = subpixel_offset2 * subpixel_offset2 * SUBPIXEL_QUALITY;

  final_offset = max(final_offset, subpixel_offset_final);

  // Read final color
  if (is_horizontal) {
    pixel_uv.y += final_offset * step_length;
  } else {
    pixel_uv.x += final_offset * step_length;
  }

  return xyz<float3>(read_image<float4>(pixels, pixel_uv));
}

}

#endif // KERNEL_ANTI_ALIASING_HPP