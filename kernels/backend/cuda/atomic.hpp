#ifndef KERNELS_BACKEND_CUDA_ATOMIC_HPP
#define KERNELS_BACKEND_CUDA_ATOMIC_HPP

namespace nova {

template <typename T>
__device__ constexpr T atomic_inc(T* address) {
  return atomicAdd(address, 1);
}

}

#endif