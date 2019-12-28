#include "intersection.cl"
#include "shading.cl"
#include "configuration.cl"

bool trace(Triangle* triangles, BVHNode* bvh, Ray* ray, float max_dist, bool fast) {
  int stack[STACK_SIZE];
  int stack_ptr = 0;
  // Push first node onto stack
  stack[++stack_ptr] = 0;

  while (stack_ptr) {
    int offset = 0;
    int num = 0;

    while (stack_ptr) {
      // Pop a node from the stack
      BVHNode node = bvh[stack[stack_ptr--]];

      if (!intersects_aabb(ray, node.top_offset_left.xyz, node.bottom_num_right.xyz)) {
        continue;
      }

      // Inner node, no triangles
      if (node.top_offset_left.w > 0) {
        // Push left and right children onto stack
        int left = node.top_offset_left.w;
        int right = node.bottom_num_right.w;

        if (right) {
          stack[++stack_ptr] = right;
        }
        if (left) {
          stack[++stack_ptr] = left;
        }
      }
      // Leaf node, no children
      else {
        offset = -node.top_offset_left.w;
        num = node.bottom_num_right.w;
        break;
      }
    }

    // If intersected, compute intersection for all triangles in the node
    for (int i = offset; i < offset + num; i++) {
      if (intersects_triangle(ray, i, triangles[i]) && fast && ray->length < max_dist) {
        return true;
      }
    }
  }

  return ray->length < max_dist;
}

kernel
void raytrace(write_only image2d_t image_out, EyeCoords ec,
              global Triangle* triangles,
              global float3* tri_normals,
              global Material* materials,
              global BVHNode* bvh) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };

  float2 alpha_beta = ec.coord_scale * (convert_float2(pixel_coords) - ec.coord_dims + 0.5f);
  float3 ray_dir = fast_normalize(alpha_beta.x * ec.eye_coord_frame0 -
                                  alpha_beta.y * ec.eye_coord_frame1 -
                                                 ec.eye_coord_frame2);
  float3 ray_pos = ec.eye_pos;

  float3 color = 0;

  Ray ray = create_ray(ray_pos, ray_dir);

  // Cast primary ray
  if (trace(triangles, bvh, &ray, FLT_MAX, false)) {
    Triangle tri = triangles[ray.intrs];
    Material mat = materials[ray.intrs];

    float3 intrs_point = ray.point + ray.direction * ray.length;

    color += mat.ambient;

    // Cast a shadow ray to the light
    float3 light_dir = LIGHT_POS - intrs_point;
    float3 normalized_light_dir = fast_normalize(light_dir);
    Ray shadow_ray = create_ray(intrs_point, normalized_light_dir);

    // Shade the pixel if ray is not blocked
    if (!trace(triangles, bvh, &shadow_ray, length(light_dir), true)) {
      float3 normal = tri_normals[ray.intrs];
      color += shade(normalized_light_dir, ray.direction, normal,
                     mat.diffuse, mat.specular, SHININESS);
    }
  }

  write_imagei(image_out, pixel_coords, convert_int4((float4)(color, 1) * 255));
}
