#include "matrix.h"

Mat3x3 mat4x3_to_mat3x3(Mat3x4 mat) {
  return {
    make_float3(mat.x.x, mat.x.y, mat.x.z),
    make_float3(mat.y.x, mat.y.y, mat.y.z),
    make_float3(mat.z.x, mat.z.y, mat.z.z),
  };
}

float3 mat3x3_vec3_mult(Mat3x3 mat, float3 vec) {
  return {
    dot(mat.x, vec),
    dot(mat.y, vec),
    dot(mat.z, vec)
  };
}

Mat3x3 mat3x3_transpose(Mat3x3 mat) {
  return {
    make_float3(mat.x.x, mat.y.x, mat.z.x),
    make_float3(mat.x.y, mat.y.y, mat.z.y),
    make_float3(mat.x.z, mat.y.z, mat.z.z)
  };
}

float3 mat4x3_vec3_mult(Mat3x4 mat, float3 vec) {
  float4 vec4 = make_float4(vec.x, vec.y, vec.z, 1.0f);
  return {
    dot(mat.x, vec4),
    dot(mat.y, vec4),
    dot(mat.z, vec4)
  };
}