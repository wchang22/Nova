#ifndef CUDA_ENTRY_H
#define CUDA_ENTRY_H

#include "constants.h"

#define STR(x) STRINGIFY(x)

#include STR(KERNELS_PATH/cuda/raytrace.h)

#endif // CUDA_ENTRY_H
