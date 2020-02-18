#include <cstddef>

#include "buffer.h"
#include "backend/cuda/types/error.h"

template <typename T>
Buffer<T>::Buffer(size_t length, const T* data) {
  CUDA_CHECK(cudaMalloc(&buffer, length * sizeof(T)))

  if (data) {
    CUDA_CHECK(cudaMemcpy(buffer, data, length * sizeof(T), cudaMemcpyHostToDevice))
  }
}

template <typename T>
Buffer<T>::~Buffer() {
  CUDA_CHECK(cudaFree(buffer))
}