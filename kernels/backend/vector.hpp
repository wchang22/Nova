#ifndef KERNELS_BACKEND_VECTOR_HPP
#define KERNELS_BACKEND_VECTOR_HPP

#include "kernels/backend/common/vector.hpp"

#if defined(KERNEL_BACKEND_OPENCL)
  #include "kernels/backend/opencl/vector.hpp"
#elif defined(KERNEL_BACKEND_CUDA)
  #include "kernels/backend/cuda/vector.hpp"
#endif

#endif