#ifndef BACKEND_ERROR_H
#define BACKEND_ERROR_H

#ifdef BACKEND_OPENCL
  #include "backend/opencl/error.h"
#elif defined(BACKEND_CUDA)
  #include "backend/cuda/error.h"
#endif

#endif // BACKEND_ERROR_H