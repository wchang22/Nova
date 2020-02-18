#ifndef BACKEND_ACCELERATOR_H
#define BACKEND_ACCELERATOR_H

#ifdef BACKEND_OPENCL
  #include "backend/opencl/accelerator/accelerator.h"
#elif defined(BACKEND_CUDA)
  #include "backend/cuda/accelerator/accelerator.h"
#endif

#endif // BACKEND_ACCELERATOR_H