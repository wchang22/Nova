#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include <cuda_runtime.h>

#include "backend/cuda/types/error.h"

template <typename T>
class Buffer {
public:
  Buffer(size_t length, const T* data = nullptr) {
    CUDA_CHECK(cudaMalloc(&buffer, length * sizeof(T)))

    if (data) {
      CUDA_CHECK(cudaMemcpy(buffer, data, length * sizeof(T), cudaMemcpyHostToDevice))
    }
  }

  ~Buffer() {
    cudaFree(buffer);
  }

  T*& data() { return buffer; };

private:
  T* buffer;
};

#endif // CUDA_BUFFER_H