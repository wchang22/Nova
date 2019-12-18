#ifndef TYPES_CL
#define TYPES_CL

typedef struct {
  float3 point;
  float3 direction;
  float length;
  int intersectable_index;
} Ray;

Ray create_ray(float3 point, float3 direction) {
  return (Ray) { point + direction * 1e-2f, direction, FLT_MAX, -1 };
}

typedef struct {
  float2 coord_scale;
  float2 coord_dims;
  float3 eye_pos;
  float3 eye_coord_frame0;
  float3 eye_coord_frame1;
  float3 eye_coord_frame2;
} EyeCoords;

typedef struct {
  float3 vertex;
  float3 normal;
  float3 edge1;
  float3 edge2;
} Triangle;

#endif // TYPES_CL