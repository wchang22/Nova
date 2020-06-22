#ifndef MATRIX_CL
#define MATRIX_CL

#include "kernel_types/matrix.hpp"

namespace nova {

inline Mat3x3 make_mat3x3(Mat3x4 mat) { return { mat.x.xyz, mat.y.xyz, mat.z.xyz }; }

inline float3 operator*(const Mat3x3& mat, float3 vec) {
  return { dot(mat.x, vec), dot(mat.y, vec), dot(mat.z, vec) };
}

inline Mat3x3 transpose(const Mat3x3& mat) {
  return { { mat.x.x, mat.y.x, mat.z.x },
           { mat.x.y, mat.y.y, mat.z.y },
           { mat.x.z, mat.y.z, mat.z.z } };
}

inline float3 operator*(const Mat3x4& mat, float3 vec) {
  float4 vec4 { vec, 1.0f };
  return { dot(mat.x, vec4), dot(mat.y, vec4), dot(mat.z, vec4) };
}

}

#endif // MATRIX_CL
