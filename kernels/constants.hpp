#ifndef KERNEL_CONSTANTS_HPP
#define KERNEL_CONSTANTS_HPP

#include "kernels/backend/math_constants.hpp"

namespace nova {

constexpr int STACK_SIZE = 64;

constexpr float RAY_EPSILON = 1e-2f; // Prevent self-shadowing

// Anti-aliasing edge thresholds
constexpr float EDGE_THRESHOLD_MIN = 0.0312f;
constexpr float EDGE_THRESHOLD_MAX = 0.125f;
constexpr uint EDGE_SEARCH_ITERATIONS = 12;
constexpr float SUBPIXEL_QUALITY = 0.75f;

}

#endif // KERNEL_CONSTANTS_HPP
