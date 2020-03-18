#ifndef CUDA_KERNEL_CONSTANTS_HPP
#define CUDA_KERNEL_CONSTANTS_HPP

#include "kernel_types/kernel_constants.hpp"
#include "math_constants.h"

__device__ __constant__ extern KernelConstants constants;

#define STACK_SIZE 96

const float RAY_EPSILON = 1e-2f; // Prevent self-shadowing
// Min epsilon to produce significant change in 8 bit colour channels
const float COLOR_EPSILON = 0.5f / 255.0f;
// Min neighbour colour difference required to raytrace instead of interpolate
const float INTERP_THRESHOLD = CUDART_SQRT_THREE_3_F;

#endif // CUDA_KERNEL_CONSTANTS_HPP
