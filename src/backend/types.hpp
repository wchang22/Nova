#ifndef BACKEND_TYPES_HPP
#define BACKEND_TYPES_HPP

#include "backend/common/types/types.hpp"

#ifdef BACKEND_OPENCL
  #include "backend/opencl/types/types.hpp"
#elif defined(BACKEND_CUDA)
  #include "backend/cuda/types/types.hpp"
#endif

#endif // BACKEND_TYPES_HPP