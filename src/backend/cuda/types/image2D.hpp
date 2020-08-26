#ifndef CUDA_IMAGE2D_HPP
#define CUDA_IMAGE2D_HPP

#include <cuda_runtime.h>

#include "backend/cuda/types/error.hpp"

namespace nova {

template <typename T>
class Image2D {
public:
  Image2D() : buffer(nullptr) {}
  Image2D(size_t width, size_t height) : width(width), height(height) {}

  Image2D(Image2D&& other) : buffer(other.buffer), width(other.width), height(other.height) {
    buffer = 0;
  }
  Image2D& operator=(Image2D&& other) {
    std::swap(buffer, other.buffer);
    width = other.width;
    height = other.height;
    return *this;
  }

  virtual ~Image2D() {}

  void copy_from(const Image2D& image) {
    CUDA_CHECK_AND_THROW(cudaMemcpy2DArrayToArray(
      buffer, 0, 0, image.buffer, 0, 0, width * sizeof(T), height, cudaMemcpyDeviceToDevice))
  }

  void fill(const T& t) {
    std::vector<T> buf(width * height, t);
    CUDA_CHECK_AND_THROW(cudaMemcpy2DToArray(buffer, 0, 0, buf.data(), width * sizeof(T),
                                             width * sizeof(T), height, cudaMemcpyHostToDevice))
  }

  std::vector<T> read() const {
    std::vector<T> image_data(width * height);
    CUDA_CHECK_AND_THROW(cudaMemcpy2DFromArray(image_data.data(), width * sizeof(T), buffer, 0, 0,
                                               width * sizeof(T), height, cudaMemcpyDeviceToHost))
    return image_data;
  }

  void write(const std::vector<T>& data) {
    CUDA_CHECK_AND_THROW(cudaMemcpy2DToArray(buffer, 0, 0, data.data(), width * sizeof(T),
                                             width * sizeof(T), height, cudaMemcpyHostToDevice))
  }

protected:
  cudaArray_t buffer;
  size_t width;
  size_t height;
};

}

#endif // CUDA_IMAGE2D_HPP