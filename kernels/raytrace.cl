kernel
void raytrace(write_only image2d_t image_out) {
  int2 global_coords = { get_global_id(0), get_global_id(1) };

  write_imagei(image_out, global_coords, (int4) 255.0);
}
