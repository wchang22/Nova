#ifndef CUDA_KERNEL_CONSTANTS_H
#define CUDA_KERNEL_CONSTANTS_H

#include "kernel_types/kernel_constants.h"

__device__ __constant__ extern KernelConstants constants;

#define STACK_SIZE 96

const float RAY_EPSILON = 1e-2f; // Prevent self-shadowing
// Min epsilon to produce significant change in 8 bit colour channels
const float COLOR_EPSILON = 0.5f / 255.0f; 

#endif // CUDA_KERNEL_CONSTANTS_H
