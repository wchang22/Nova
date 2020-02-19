#ifndef MATRIX_H
#define MATRIX_H

#include "vector_math.h"

struct Mat3x3 {
  float3 x;
  float3 y;
  float3 z;
};

struct Mat4x3 {
  float4 x;
  float4 y;
  float4 z;
};

Mat3x3 mat4x3_to_mat3x3(Mat4x3 mat);
float3 mat3x3_vec3_mult(Mat3x3 mat, float3 vec);
Mat3x3 mat3x3_transpose(Mat3x3 mat);
float3 mat4x3_vec3_mult(Mat4x3 mat, float3 vec);

#endif // MATRIX_H
