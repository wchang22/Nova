#ifndef BACKEND_TYPES_H
#define BACKEND_TYPES_H

#ifdef BACKEND_OPENCL
  #include "backend/opencl/types/types.h"
#elif defined(BACKEND_CUDA)
  #include "backend/cuda/types/types.h"
#endif

#endif // BACKEND_TYPES_H