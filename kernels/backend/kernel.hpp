#ifndef KERNELS_BACKEND_KERNEL_HPP
#define KERNELS_BACKEND_KERNEL_HPP

#if defined(KERNEL_BACKEND_OPENCL)
  #include "kernels/backend/opencl/kernel.hpp"
#elif defined(KERNEL_BACKEND_CUDA)
  #include "kernels/backend/cuda/kernel.hpp"
#endif

#endif