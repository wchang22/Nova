#ifndef TYPES_CL
#define TYPES_CL

#include "configuration.cl"

typedef struct {
  float3 point;
  float3 direction;
  float3 inv_direction;
  float length;
  int intrs;
} Ray;

Ray create_ray(float3 point, float3 direction) {
  return (Ray) { point + direction * RAY_EPSILON, direction, 1.0f / direction, FLT_MAX, -1 };
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
  float3 edge1;
  float3 edge2;
} Triangle;

typedef struct {
  float3 ambient;
  float3 diffuse;
  float3 specular;
} Material;

// Packed to 32 bytes to fit in a cache line
// The 4th element of top_offset_left contains either the triangle offset or the left index
// The 4th element of bottom_num_right contains either the number of triangles or the right index
// depending on whether or not the node is an inner node or a leaf node
// If the 4th element of top_offset_left < 0, then the node is a leaf node
typedef struct {
  float4 top_offset_left;
  float4 bottom_num_right;
} BVHNode;

#endif // TYPES_CL
