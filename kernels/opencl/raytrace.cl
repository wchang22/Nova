#include "anti_aliasing.cl"
#include "constants.cl"
#include "intersection.cl"
#include "texture.cl"
#include "transforms.cl"

bool find_intersection(
  global Triangle* triangles, global BVHNode* bvh, Ray ray, Intersection* min_intrs, bool fast) {
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
      BVHNode node = bvh[node_index];

      if (!intersects_aabb(ray, node.top_offset_left.xyz, node.bottom_num_right.xyz)) {
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
        // Pop node off stack
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

  return min_intrs->tri_index != -1;
}

float3 trace_ray(int2 pixel_coords,
                 SceneParams scene_params,
                 global Triangle* triangles,
                 global TriangleMeta* tri_meta,
                 global BVHNode* bvh,
                 read_only image2d_array_t materials,
                 read_only image2d_t sky) {
  float2 alpha_beta = scene_params.eye_coords.coord_scale *
                      (convert_float2(pixel_coords) - scene_params.eye_coords.coord_dims + 0.5f);
  float3 ray_dir = fast_normalize(alpha_beta.x * scene_params.eye_coords.eye_coord_frame.x -
                                  alpha_beta.y * scene_params.eye_coords.eye_coord_frame.y -
                                  scene_params.eye_coords.eye_coord_frame.z);
  float3 ray_pos = scene_params.eye_coords.eye_pos;

  float3 color = 0.0f;
  float3 reflectance = 1.0f;

  for (int depth = 0; depth < scene_params.ray_bounces; depth++) {
    Ray ray = create_ray(ray_pos, ray_dir, RAY_EPSILON);

    Intersection intrs = NO_INTERSECTION;

    // Cast primary/reflection ray
    if (!find_intersection(triangles, bvh, ray, &intrs, false)) {
      // TODO: IBL instead of just skymap
      if (depth == 0) {
        color = read_sky(sky, ray_dir);
      }
      break;
    }

    TriangleMeta meta = tri_meta[intrs.tri_index];

    // Calculate intersection point
    float3 intrs_point = ray.origin + ray.direction * intrs.length;

    // Interpolate texture coords from vertex data
    float2 texture_coord = triangle_interpolate2(intrs.barycentric, meta.texture_coord1,
                                                 meta.texture_coord2, meta.texture_coord3);

    // Look up materials
    // clang-format off
    float3 diffuse = read_material(materials, meta, texture_coord,
      meta.diffuse_index, scene_params.shading_diffuse) * meta.kD;
    float metallic = read_material(materials, meta, texture_coord,
      meta.metallic_index, scene_params.shading_metallic).x;
    float roughness = read_material(materials, meta, texture_coord,
      meta.roughness_index, scene_params.shading_roughness).x;
    float ambient_occlusion = read_material(materials, meta, texture_coord,
      meta.ambient_occlusion_index, scene_params.shading_ambient_occlusion).x;
    // clang-format on

    float3 normal = compute_normal(materials, meta, texture_coord, intrs.barycentric);

    // Add ambient color even if pixel is in shadow
    float3 intrs_color = diffuse * ambient_occlusion * 0.03f * meta.kA + meta.kE;

    // Calculate lighting params
    float3 light_dir = fast_normalize(scene_params.light_position - intrs_point);
    float3 view_dir = -ray.direction;
    float3 half_dir = fast_normalize(light_dir + view_dir);
    float light_distance = fast_distance(scene_params.light_position, intrs_point);
    float3 kS = specularity(view_dir, half_dir, diffuse, metallic) * meta.kS;

    float3 local_illum = shade(light_dir, view_dir, half_dir, scene_params.light_intensity,
                               light_distance, normal, diffuse, kS, metallic, roughness);

    // Only cast a shadow ray if it will produce a color change
    if (any(isgreaterequal(local_illum, COLOR_EPSILON))) {
      // Cast a shadow ray to the light
      Ray shadow_ray = create_ray(intrs_point, light_dir, RAY_EPSILON);
      Intersection light_intrs = NO_INTERSECTION;
      // Ensure objects blocking light are not behind the light
      light_intrs.length = light_distance;

      // Shade the pixel if ray is not blocked
      if (!find_intersection(triangles, bvh, shadow_ray, &light_intrs, true)) {
        intrs_color += local_illum;
      }
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

  return gamma_correct(tone_map(color, scene_params.exposure));
}

kernel void kernel_raytrace(SceneParams scene_params,
                            write_only image2d_t temp_pixels1,
                            write_only image2d_t temp_pixels2,
                            uint2 pixel_dims,
                            global Triangle* triangles,
                            global TriangleMeta* tri_meta,
                            global BVHNode* bvh,
                            read_only image2d_array_t materials,
                            read_only image2d_t sky) {
  int2 packed_pixel_coords = { get_global_id(0), get_global_id(1) };
  if (packed_pixel_coords.x >= pixel_dims.x && packed_pixel_coords.y >= pixel_dims.y / 2) {
    return;
  }

  int2 pixel_coords = packed_pixel_coords;
  pixel_coords.y = 2 * pixel_coords.y + (pixel_coords.x & 1);

  float3 color = trace_ray(pixel_coords, scene_params, triangles, tri_meta, bvh, materials, sky);
  // Use a packed uchar4 image to save memory and bandwidth
  write_imageui(temp_pixels1, packed_pixel_coords, (uint4)(float3_to_uint3(color), 255));
  write_imagef(temp_pixels2, pixel_coords, (float4)(color, 1.0f));
}

kernel void kernel_interpolate(read_only image2d_t temp_pixels1,
                               write_only image2d_t temp_pixels2,
                               uint2 pixel_dims,
                               global uint* rem_pixels_counter,
                               global int2* rem_coords) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y / 2) {
    return;
  }
  pixel_coords.y = 2 * pixel_coords.y + 1 - (pixel_coords.x & 1);

  // Sample 4 neighbours
  const int2 neighbor_offsets[] = { { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 } };
  uint3 neighbors[4];
  for (uint i = 0; i < 4; i++) {
    // Lookup from the packed uchar4 image
    int2 packed_coords = pixel_coords + neighbor_offsets[i];
    packed_coords.y = (packed_coords.y - (packed_coords.x & 1)) / 2;
    neighbors[i] = read_imageui(temp_pixels1, image_sampler, packed_coords).xyz;
  }

  // Check color differences in the neighbours
  uint3 color_max = max(neighbors[0], max(neighbors[1], max(neighbors[2], neighbors[3])));
  uint3 color_min = min(neighbors[0], min(neighbors[1], min(neighbors[2], neighbors[3])));
  float3 color_range = uint3_to_float3(color_max - color_min);

  // If difference is large, store coords to raytrace later
  if (length(color_range) > INTERP_THRESHOLD) {
    rem_coords[atomic_inc(rem_pixels_counter)] = pixel_coords;
  }
  // Otherwise, interpolate
  else {
    uint3 color = (neighbors[0] + neighbors[1] + neighbors[2] + neighbors[3]) / 4;
    write_imagef(temp_pixels2, pixel_coords, (float4)(uint3_to_float3(color), 1.0f));
  }
}

kernel void kernel_fill_remaining(SceneParams scene_params,
                                  write_only image2d_t temp_pixels2,
                                  uint2 pixel_dims,
                                  global Triangle* triangles,
                                  global TriangleMeta* tri_meta,
                                  global BVHNode* bvh,
                                  read_only image2d_array_t materials,
                                  read_only image2d_t sky,
                                  global uint* rem_pixels_counter,
                                  global int2* rem_coords) {
  uint id = get_global_id(0);
  if (id >= *rem_pixels_counter) {
    return;
  }
  int2 pixel_coords = rem_coords[id];

  float3 color = trace_ray(pixel_coords, scene_params, triangles, tri_meta, bvh, materials, sky);
  write_imagef(temp_pixels2, pixel_coords, (float4)(color, 1.0f));
}

kernel void kernel_post_process(SceneParams scene_params,
                                read_only image2d_t temp_pixels2,
                                write_only image2d_t pixels,
                                uint2 pixel_dims) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };
  if (pixel_coords.x >= pixel_dims.x && pixel_coords.y >= pixel_dims.y) {
    return;
  }

  float2 inv_pixel_dims = 1.0f / convert_float2(pixel_dims);
  float2 pixel_uv = (convert_float2(pixel_coords) + 0.5f) * inv_pixel_dims;

  float3 color;
  if (scene_params.anti_aliasing) {
    color = fxaa(temp_pixels2, inv_pixel_dims, pixel_uv);
  } else {
    color = read_imagef(temp_pixels2, image_sampler_norm, pixel_uv).xyz;
  }

  write_imageui(pixels, pixel_coords, (uint4)(float3_to_uint3(color), 255));
}