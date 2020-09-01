#ifndef KERNELS_BACKEND_CUDA_KERNEL_HPP
#define KERNELS_BACKEND_CUDA_KERNEL_HPP

#define DEVICE __device__
#define HOST __host__
#define KERNEL __global__
#define GLOBAL
#define LOCAL __shared__
#define CONSTANT

#include <cstdio>

namespace nova {

__device__ inline int get_global_id(int i) {
  switch (i) {
    case 0:
      return blockDim.x * blockIdx.x + threadIdx.x;
    case 1:
      return blockDim.y * blockIdx.y + threadIdx.y;
    case 2:
      return blockDim.z * blockIdx.z + threadIdx.z;
    default:
      return 0;
  }
}

}

#endif