#include "intersection.cl"
#include "shading.cl"

bool compute_intersection(global Triangle* triangles, int num_triangles, Ray* ray) {
  for (int i = 0; i < num_triangles; i++) {
    intersects(ray, i, triangles[i]);
  }

  return ray->intrs != -1;
}

kernel
void raytrace(write_only image2d_t image_out, EyeCoords ec,
              global Triangle* triangles, int num_triangles) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };

  float2 alpha_beta = ec.coord_scale * (convert_float2(pixel_coords) - ec.coord_dims + 0.5f);
  float3 ray_dir = normalize(alpha_beta.x * ec.eye_coord_frame0 -
                             alpha_beta.y * ec.eye_coord_frame1 -
                                            ec.eye_coord_frame2);
  float3 ray_pos = ec.eye_pos;

  float3 color = 0;

  Ray ray = create_ray(ray_pos, ray_dir);

  if (compute_intersection(triangles, num_triangles, &ray)) {
    float3 intrs_point = ray.point + ray.direction * ray.length;
    color += shade(intrs_point, ray.direction, normalize(triangles[ray.intrs].normal),
                   (float3)(0.4, 0.4, 0.4), (float3)(0.7, 0.7, 0.7), 32);
  }

  write_imagei(image_out, pixel_coords, convert_int4((float4)(color, 1) * 255));
}
