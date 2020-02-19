#ifndef CUDA_IMAGE2D_H
#define CUDA_IMAGE2D_H

#include "backend/cuda/types/flags.h"

template<typename T>
class Image2D {
public:
  Image2D(AddressMode address_mode, FilterMode filter_mode, bool normalized_coords,
          size_t width, size_t height, T* data = nullptr);
  ~Image2D();

  cudaTextureObject_t& data() { return tex; };
  const T* ptr() const { return buffer; };

private:
  cudaTextureObject_t tex;
  T* buffer;
};

#endif // CUDA_IMAGE2D_H