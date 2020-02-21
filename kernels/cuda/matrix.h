#ifndef CUDA_KERNEL_MATRIX_H
#define CUDA_KERNEL_MATRIX_H

#include "vector_math.h"
#include "kernel_types/matrix.h"

Mat3x3 mat4x3_to_mat3x3(Mat3x4 mat);
float3 mat3x3_vec3_mult(Mat3x3 mat, float3 vec);
Mat3x3 mat3x3_transpose(Mat3x3 mat);
float3 mat4x3_vec3_mult(Mat3x4 mat, float3 vec);

#endif // CUDA_KERNEL_MATRIX_H
