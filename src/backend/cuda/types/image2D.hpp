#ifndef CUDA_IMAGE2D_HPP
#define CUDA_IMAGE2D_HPP

#include <cuda_runtime.h>

#include "backend/cuda/types/error.hpp"

namespace nova {

template <typename T>
class Image2D {
public:
  Image2D(size_t width, size_t height) : width(width), height(height) {}

  virtual ~Image2D() {}

  void copy_from(const Image2D& image) {
    CUDA_CHECK_AND_THROW(cudaMemcpy2DArrayToArray(
      buffer, 0, 0, image.buffer, 0, 0, width * sizeof(T), height, cudaMemcpyDeviceToDevice))
  }

protected:
  cudaArray_t buffer;
  size_t width;
  size_t height;
};

}

#endif // CUDA_IMAGE2D_HPP