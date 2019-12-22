#include "intersection.cl"
#include "shading.cl"

const int SHININESS = 32;
const int STACK_SIZE = 30;

bool compute_intersection(Triangle* triangles, int num_triangles, BVHNode* bvh, Ray* ray) {
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
      intersects(ray, i, triangles[i]);
    }
  }

  return ray->intrs != -1;
}

kernel
void raytrace(write_only image2d_t image_out, EyeCoords ec,
              global Triangle* triangles, int num_triangles,
              global BVHNode* bvh) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };

  float2 alpha_beta = ec.coord_scale * (convert_float2(pixel_coords) - ec.coord_dims + 0.5f);
  float3 ray_dir = normalize(alpha_beta.x * ec.eye_coord_frame0 -
                             alpha_beta.y * ec.eye_coord_frame1 -
                                            ec.eye_coord_frame2);
  float3 ray_pos = ec.eye_pos;

  float3 color = 0;

  Ray ray = create_ray(ray_pos, ray_dir);

  if (compute_intersection(triangles, num_triangles, bvh, &ray)) {
    float3 intrs_point = ray.point + ray.direction * ray.length;
    Triangle tri = triangles[ray.intrs];
    color += tri.ambient;
    color += shade(intrs_point, ray.direction, normalize(tri.normal),
                   tri.diffuse, tri.specular, SHININESS);
  }

  write_imagei(image_out, pixel_coords, convert_int4((float4)(color, 1) * 255));
}
