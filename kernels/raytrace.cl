#include "intersection.cl"
#include "shading.cl"
#include "configuration.cl"
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

  // Push first node onto stack front
  stack[++node_ptr] = 0;

  while (node_ptr >= 0) {
    // Make sure tri_ptr and node_ptr do not collide
    while (node_ptr >= 0 && tri_ptr > node_ptr + 2) {
      // Pop a node from the stack front
      BVHNode node = bvh[stack[node_ptr--]];

      if (!intersects_aabb(ray, node.top_offset_left.xyz, node.bottom_num_right.xyz)) {
        continue;
      }

      // Inner node, no triangles
      if (node.top_offset_left.w > 0) {
        // Push left and right children onto stack front
        uint left = node.top_offset_left.w;
        uint right = node.bottom_num_right.w;

        if (right) {
          stack[++node_ptr] = right;
        }
        if (left) {
          stack[++node_ptr] = left;
        }
      }
      // Leaf node, no children
      else {
        uint offset = -node.top_offset_left.w;
        uint num = node.bottom_num_right.w;

        // Pack offset and num into a single uint to save memory
        uint packed_triangle_data = (offset & TRIANGLE_OFFSET_MASK) | (num << TRIANGLE_NUM_SHIFT);

        // Push list of triangles to stack back
        stack[--tri_ptr] = packed_triangle_data;
      }
    }

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
  }

  return min_intrs->tri_index != -1;
}

kernel
void raytrace(write_only image2d_t image_out, EyeCoords ec,
              global Triangle* triangles,
              global TriangleMeta* tri_meta,
              global BVHNode* bvh,
              read_only image2d_array_t materials) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };

  float2 alpha_beta = ec.coord_scale * (convert_float2(pixel_coords) - ec.coord_dims + 0.5f);
  float3 ray_dir = fast_normalize(alpha_beta.x * ec.eye_coord_frame.x -
                                  alpha_beta.y * ec.eye_coord_frame.y -
                                                 ec.eye_coord_frame.z);
  float3 ray_pos = ec.eye_pos;

  float3 color = 0;
  float3 reflectance = 1;

  for (int depth = 0; depth < RAY_RECURSION_DEPTH; depth++) {
    Ray ray = create_ray(ray_pos, ray_dir, RAY_EPSILON);

    Intersection intrs = NO_INTERSECTION;

    // Cast primary/reflection ray
    if (!trace(triangles, bvh, ray, &intrs, false)) {
      break;
    }

    TriangleMeta meta = tri_meta[intrs.tri_index];

    // Calculate intersection point
    intrs.point = ray.origin + ray.direction * intrs.length;

    // Interpolate triangle normal and texture coords from vertex data
    float3 normal = fast_normalize(
      triangle_interpolate3(intrs.barycentric, meta.normal1, meta.normal2, meta.normal3)
    );
    float2 texture_coord = triangle_interpolate2(
      intrs.barycentric, meta.texture_coord1, meta.texture_coord2, meta.texture_coord3
    );

    // Look up materials
    float3 ambient =
      read_material(materials, meta, texture_coord, meta.ambient_index, DEFAULT_AMBIENT);
    float3 diffuse =
      read_material(materials, meta, texture_coord, meta.diffuse_index, DEFAULT_DIFFUSE);
    float3 specular =
      read_material(materials, meta, texture_coord, meta.specular_index, DEFAULT_SPECULAR);

    // Add ambient color even if pixel is in shadow
    float3 intrs_color = ambient;

    // Cast a shadow ray to the light
    float3 light_dir = fast_normalize(LIGHT_POS - intrs.point);
    float light_distance = fast_distance(LIGHT_POS, intrs.point);
    Ray shadow_ray = create_ray(intrs.point, light_dir, RAY_EPSILON);

    Intersection light_intrs = NO_INTERSECTION;
    // Ensure objects blocking light are not behind the light
    light_intrs.length = light_distance;

    // Shade the pixel if ray is not blocked
    if (!trace(triangles, bvh, shadow_ray, &light_intrs, true)) {
      intrs_color += shade(light_dir, ray.direction, normal, diffuse, specular, SHININESS);
    }

    /*
     * Normally, color is calculated recursively:
     * (intrs_color + specular * (intrs_color of reflected ray))
     * So we use an additional "reflectance" value to unroll the recursion
     */
    color += reflectance * intrs_color;
    reflectance *= specular;

    // Reflect ray off of intersection point
    ray_pos = intrs.point;
    ray_dir = reflect(ray_dir, normal);
  }

  write_imageui(image_out, pixel_coords, convert_uint4((float4)(color, 1) * 255));
}
