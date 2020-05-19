#ifndef CUDA_KERNEL_CONSTANTS_HPP
#define CUDA_KERNEL_CONSTANTS_HPP

#include "kernel_types/kernel_constants.hpp"
#include "kernel_types/scene_params.hpp"
#include "math_constants.h"

namespace nova {

__device__ __constant__ extern KernelConstants constants;
__device__ __constant__ extern SceneParams params;

#define STACK_SIZE 96

const float RAY_EPSILON = 1e-2f; // Prevent self-shadowing
// Min epsilon to produce significant change in 8 bit colour channels
const float COLOR_EPSILON = 0.5f / 255.0f;
// Min neighbour colour difference required to raytrace instead of interpolate
const float INTERP_THRESHOLD = CUDART_SQRT_THREE_3_F;

// Anti-aliasing edge thresholds
const float EDGE_THRESHOLD_MIN = 0.0312f;
const float EDGE_THRESHOLD_MAX = 0.125f;
const uint EDGE_SEARCH_ITERATIONS = 12;
const float SUBPIXEL_QUALITY = 0.75f;

}

#endif // CUDA_KERNEL_CONSTANTS_HPP
