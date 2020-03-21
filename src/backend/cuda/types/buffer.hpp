#ifndef CUDA_BUFFER_HPP
#define CUDA_BUFFER_HPP

#include <cuda_runtime.h>
#include <vector>

#include "backend/cuda/types/error.hpp"

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

  void write(const std::vector<T>& v) {
    CUDA_CHECK(cudaMemcpy(buffer, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice))
  }

  void write(const T& t) {
    CUDA_CHECK(cudaMemcpy(buffer, &t, sizeof(T), cudaMemcpyHostToDevice))
  }

  std::vector<T> read(size_t length) const {
    std::vector<T> buf(length);
    CUDA_CHECK(cudaMemcpy(buf.data(), buffer, length * sizeof(T), cudaMemcpyDeviceToHost))
    return buf;
  }

  T read() const {
    T t;
    CUDA_CHECK(cudaMemcpy(&t, buffer, sizeof(T), cudaMemcpyDeviceToHost))
    return t;
  }

  T*& data() { return buffer; };
  const T*& data() const { return buffer; };

private:
  T* buffer;
  size_t length;
};

#endif // CUDA_BUFFER_HPP