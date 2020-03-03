#include "intersection.cl"
#include "texture.cl"
#include "constants.cl"
#include "transforms.cl"

bool trace(global Triangle* triangles, global BVHNode* bvh, Ray ray, Intersection* min_intrs,
           bool fast) {
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
        uint packed_triangle_data = (offset & TRIANGLE_OFFSET_MASK) | (num << TRIANGLE_NUM_SHIFT);

        // Push list of triangles to stack back
        stack[--tri_ptr] = packed_triangle_data;

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

kernel
void kernel_generate_rays(
  // Stage outputs
  global PackedRay* rays, global float3* colors, global float3* reflectances,
  // Stage read-only
  EyeCoords ec, uint image_width
) {
  int id = get_global_linear_id();
  int2 pixel_coords = { id % image_width, id / image_width };

  float2 alpha_beta = ec.coord_scale * (convert_float2(pixel_coords) - ec.coord_dims + 0.5f);
  float3 ray_dir = fast_normalize(alpha_beta.x * ec.eye_coord_frame.x -
                                  alpha_beta.y * ec.eye_coord_frame.y -
                                                 ec.eye_coord_frame.z);
  float3 ray_pos = ec.eye_pos;

  rays[id] = pack_ray(create_ray(ray_pos, ray_dir, id, RAY_EPSILON));
  colors[id] = 0.0f;
  reflectances[id] = 1.0f;
}

kernel
void kernel_intersect_rays(
  // Stage inputs
  global PackedRay* rays,
  // Stage outputs
  global Intersection* intersections, global uint* global_count,
  // Stage read-only
  global Triangle* triangles, global BVHNode* bvh
) {
  int id = get_global_linear_id();

  Ray ray = unpack_ray(rays[id]);
  Intersection intrs = create_intersection(id);

  if (trace(triangles, bvh, ray, &intrs, false)) {
    uint intrs_id = atomic_inc(global_count);
    intersections[intrs_id] = intrs;
  }
}

kernel
void kernel_shade_pixels(
  // Stage inputs
  global PackedRay* rays, global float3* colors, global float3* reflectances,
  global Intersection* intersections,
  // Stage outputs
  global PackedRay* reflection_rays, global uint* global_count,
  // Stage read-only
  global Triangle* triangles, global BVHNode* bvh,
  global TriangleMeta* tri_meta, read_only image2d_array_t materials
) {
  int id = get_global_linear_id();

  Intersection intrs = intersections[id];
  Ray ray = unpack_ray(rays[intrs.ray_index]);
  TriangleMeta meta = tri_meta[intrs.tri_index];
  float3 reflectance = reflectances[ray.image_index];

  // Calculate intersection point
  float3 intrs_point = ray.origin + ray.direction * intrs.length;

  // Interpolate texture coords from vertex data
  float2 texture_coord = triangle_interpolate2(
    intrs.barycentric, meta.texture_coord1, meta.texture_coord2, meta.texture_coord3
  );

  // Look up materials
  float3 diffuse =
    read_material(materials, meta, texture_coord, meta.diffuse_index, DEFAULT_DIFFUSE);
  float metallic =
    read_material(materials, meta, texture_coord, meta.metallic_index, DEFAULT_METALLIC).x;
  float roughness =
    read_material(materials, meta, texture_coord, meta.roughness_index, DEFAULT_ROUGHNESS).x;
  float ambient_occlusion =
    read_material(materials, meta, texture_coord,
                  meta.ambient_occlusion_index, DEFAULT_AMBIENT_OCCLUSION).x;

  float3 normal = compute_normal(materials, meta, texture_coord, intrs.barycentric);

  // Add ambient color even if pixel is in shadow
  float3 intrs_color = diffuse * ambient_occlusion * 0.03f;

  // Calculate lighting params
  float3 light_dir = fast_normalize(LIGHT_POSITION - intrs_point);
  float3 view_dir = -ray.direction;
  float3 half_dir = fast_normalize(light_dir + view_dir);
  float light_distance = fast_distance(LIGHT_POSITION, intrs_point);
  float3 kS = specularity(view_dir, half_dir, diffuse, metallic);

  // Cast a shadow ray to the light
  Ray shadow_ray = create_ray(intrs_point, light_dir, 0, RAY_EPSILON);
  Intersection light_intrs = create_intersection(0);
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
  colors[ray.image_index] += reflectance * intrs_color;
  reflectances[ray.image_index] *= kS;

  // Stop if reflectance is too low to produce a color change
  if (!all(isless(reflectance, COLOR_EPSILON))) {
    uint ray_id = atomic_inc(global_count);

    // Reflect ray off of intersection point
    float3 ray_pos = intrs_point;
    float3 ray_dir = reflect(ray.direction, normal);
    reflection_rays[ray_id] =
      pack_ray(create_ray(intrs_point, ray_dir, ray.image_index, RAY_EPSILON));
  }
}

kernel
void kernel_fill_image(
  // Stage inputs
  global float3* colors,
  // Stage write-only
  write_only image2d_t image_out
) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };
  int id = get_global_linear_id();
  float3 color = colors[id];

  write_imageui(image_out, pixel_coords, convert_uint4((float4)(color, 1.0f) * 255.0f));
}
