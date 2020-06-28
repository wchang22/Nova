#ifndef KERNELS_BACKEND_MATH_CONSTANTS_HPP
#define KERNELS_BACKEND_MATH_CONSTANTS_HPP

#if defined(KERNEL_BACKEND_OPENCL)

#elif defined(KERNEL_BACKEND_CUDA)
  #include "kernels/backend/cuda/math_constants.hpp"
#endif

#endif