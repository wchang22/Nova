#ifndef KERNEL_CONSTANTS_HPP
#define KERNEL_CONSTANTS_HPP

#include "kernel_types/scene_params.hpp"
#include "kernels/backend/math_constants.hpp"
#include "shared_constants.hpp"

namespace nova {

constexpr int STACK_SIZE = 96;

constexpr float RAY_EPSILON = 1e-2f; // Prevent self-shadowing
// Min epsilon to produce significant change in 8 bit colour channels
constexpr float COLOR_EPSILON = 0.5f / 255.0f;
// Min neighbour colour difference required to raytrace instead of interpolate
constexpr float INTERP_THRESHOLD = M_SQRT3_3F;

// Anti-aliasing edge thresholds
constexpr float EDGE_THRESHOLD_MIN = 0.0312f;
constexpr float EDGE_THRESHOLD_MAX = 0.125f;
constexpr uint EDGE_SEARCH_ITERATIONS = 12;
constexpr float SUBPIXEL_QUALITY = 0.75f;

}

#endif // KERNEL_CONSTANTS_HPP
