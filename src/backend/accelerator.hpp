#ifndef BACKEND_ACCELERATOR_HPP
#define BACKEND_ACCELERATOR_HPP

#ifdef BACKEND_OPENCL
  #include "backend/opencl/accelerator/accelerator.hpp"
#elif defined(BACKEND_CUDA)
  #include "backend/cuda/accelerator/accelerator.hpp"
#endif

#endif // BACKEND_ACCELERATOR_HPP