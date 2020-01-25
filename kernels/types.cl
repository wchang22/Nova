#ifndef TYPES_CL
#define TYPES_CL

#include "configuration.cl"
#include "matrix.cl"

typedef struct {
  float3 origin;
  float3 direction;
} Ray;

Ray create_ray(float3 point, float3 direction, float epsilon) {
  return (Ray) { point + direction * epsilon, direction };
}

typedef struct {
  float3 point;
  float3 barycentric;
  float length;
  int tri_index;
} Intersection;

constant Intersection NO_INTERSECTION = {
  (float3) 0, (float3) 0, FLT_MAX, -1
};

typedef struct {
  float2 coord_scale;
  float2 coord_dims;
  float3 eye_pos;
  Mat3x3 eye_coord_frame;
} EyeCoords;

// Woop 4x3 affine transform matrix
typedef struct {
  Mat4x3 transform;
} Triangle;

// We look up the triangle metadata and material separately to reduce cache pressure
typedef struct {
  float3 normal1;
  float3 normal2;
  float3 normal3;
  float3 tangent1;
  float3 tangent2;
  float3 tangent3;
  float3 bitangent1;
  float3 bitangent2;
  float3 bitangent3;
  float2 texture_coord1;
  float2 texture_coord2;
  float2 texture_coord3;
  int ambient_index;
  int diffuse_index;
  int specular_index;
  int normal_index;
} TriangleMeta;

// Packed to 32 bytes to fit in a cache line
// The 4th element of top_offset_left contains either the triangle offset or the left index
// The 4th element of bottom_num_right contains either the number of triangles or the right index
// depending on whether or not the node is an inner node or a leaf node
// If the 4th element of bottom_num_right < 0, then the node is a leaf node
typedef struct {
  float4 top_offset_left;
  float4 bottom_num_right;
} BVHNode;

#endif // TYPES_CL
