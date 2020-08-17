#ifndef KERNELS_BACKEND_ASSERTION_HPP
#define KERNELS_BACKEND_ASSERTION_HPP

#if defined(KERNEL_BACKEND_OPENCL)
  #include "kernels/backend/opencl/assertion.hpp"
#elif defined(KERNEL_BACKEND_CUDA)
  #include <cassert>
#endif

#endif