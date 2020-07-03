#ifndef KERNELS_BACKEND_IMAGE_HPP
#define KERNELS_BACKEND_IMAGE_HPP

#if defined(KERNEL_BACKEND_OPENCL)
  #include "kernels/backend/opencl/image.hpp"
#elif defined(KERNEL_BACKEND_CUDA)
  #include "kernels/backend/cuda/image.hpp"
#endif

#endif