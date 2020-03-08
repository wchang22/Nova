#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include <cuda_runtime.h>
#include <vector>

#include "backend/cuda/types/error.h"

template <typename T>
class Buffer {
public:
  Buffer(size_t length, const T* data = nullptr)
    : length(length)
  {
    CUDA_CHECK(cudaMalloc(&buffer, length * sizeof(T)))

    if (data) {
      CUDA_CHECK(cudaMemcpy(buffer, data, length * sizeof(T), cudaMemcpyHostToDevice))
    }
  }

  ~Buffer() {
    cudaFree(buffer);
  }

  void fill(size_t length, const T& t) {
    std::vector<T> buf(length, t);
    CUDA_CHECK(cudaMemcpy(buffer, buf.data(), length * sizeof(T), cudaMemcpyHostToDevice))
  }

  std::vector<T> read(size_t length) const {
    std::vector<T> buf(length);
    CUDA_CHECK(cudaMemcpy(buf.data(), buffer, length * sizeof(T), cudaMemcpyDeviceToHost))
    return buf;
  }

  T*& data() { return buffer; };
  const T*& data() const { return buffer; };

private:
  T* buffer;
  size_t length;
};

#endif // CUDA_BUFFER_H