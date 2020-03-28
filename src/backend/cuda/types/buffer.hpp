#ifndef CUDA_BUFFER_HPP
#define CUDA_BUFFER_HPP

#include <cuda_runtime.h>
#include <vector>

#include "backend/cuda/types/error.hpp"

template <typename T>
class Buffer {
public:
  Buffer() : buffer(nullptr), length(0) {}

  Buffer(size_t length, const T* data = nullptr) : length(length) {
    CUDA_CHECK_AND_THROW(cudaMalloc(&buffer, length * sizeof(T)))

    if (data) {
      CUDA_CHECK_AND_THROW(cudaMemcpy(buffer, data, length * sizeof(T), cudaMemcpyHostToDevice))
    }
  }

  ~Buffer() { CUDA_CHECK(cudaFree(buffer)) }

  // Buffer(const Buffer& other) : length(other.length) {
  //   CUDA_CHECK_AND_THROW(cudaMalloc(&buffer, length * sizeof(T)))
  //   if (other.buffer) {
  //     CUDA_CHECK_AND_THROW(cudaMemcpy(buffer, other.buffer, length * sizeof(T),
  //     cudaMemcpyDeviceToDevice))
  //   }
  // }
  Buffer(Buffer&& other)
    : length(other.length),
  buffer(other.buffer) {
    other.length = 0;
    other.buffer = nullptr;
  }
  // Buffer& operator=(const Buffer& other) {
  //   length = other.length;
  //   cudaFree(buffer);
  //   CUDA_CHECK_AND_THROW(cudaMalloc(&buffer, length * sizeof(T)))
  //   if (other.buffer) {
  //     CUDA_CHECK_AND_THROW(cudaMemcpy(buffer, other.buffer, length * sizeof(T),
  //     cudaMemcpyDeviceToDevice))
  //   }
  //   return *this;
  // }
  Buffer& operator=(Buffer&& other) {
    std::swap(length, other.length);
    std::swap(buffer, other.buffer);
    return *this;
  }

  void fill(size_t length, const T& t) {
    std::vector<T> buf(length, t);
    CUDA_CHECK_AND_THROW(cudaMemcpy(buffer, buf.data(), length * sizeof(T), cudaMemcpyHostToDevice))
  }

  void write(const std::vector<T>& v) {
    CUDA_CHECK_AND_THROW(cudaMemcpy(buffer, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice))
  }

  void write(const T& t) {
    CUDA_CHECK_AND_THROW(cudaMemcpy(buffer, &t, sizeof(T), cudaMemcpyHostToDevice))
  }

  std::vector<T> read(size_t length) const {
    std::vector<T> buf(length);
    CUDA_CHECK_AND_THROW(cudaMemcpy(buf.data(), buffer, length * sizeof(T), cudaMemcpyDeviceToHost))
    return buf;
  }

  T read() const {
    T t;
    CUDA_CHECK_AND_THROW(cudaMemcpy(&t, buffer, sizeof(T), cudaMemcpyDeviceToHost))
    return t;
  }

  T*& data() { return buffer; };
  const T*& data() const { return buffer; };

private:
  T* buffer;
  size_t length;
};

#endif // CUDA_BUFFER_HPP