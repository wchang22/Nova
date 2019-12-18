#include "intersection.cl"

kernel
void raytrace(write_only image2d_t image_out, EyeCoords ec,
              global void* triangle_data, int num_triangles) {
  int2 pixel_coords = { get_global_id(0), get_global_id(1) };
  Triangle* triangles = triangle_data;

  float2 alpha_beta = ec.coord_scale * (convert_float2(pixel_coords) - ec.coord_dims + 0.5f);
  float3 ray_dir = normalize(alpha_beta.x * ec.eye_coord_frame0 -
                             alpha_beta.y * ec.eye_coord_frame1 -
                                            ec.eye_coord_frame2);
  float3 ray_pos = ec.eye_pos;

  float3 color = 0;

  Ray ray = create_ray(ray_pos, ray_dir);

  for (int i = 0; i < num_triangles; i++) {
    if (intersects(&ray, i, triangles[i])) {
      color = (float3) {1, 0, 0};
    }
  }

  write_imagei(image_out, pixel_coords, convert_int4((float4)(color, 1) * 255));
}
