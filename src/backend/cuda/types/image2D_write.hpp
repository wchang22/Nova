#ifndef CUDA_IMAGE2D_WRITE_HPP
#define CUDA_IMAGE2D_WRITE_HPP

#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "backend/cuda/types/error.hpp"
#include "backend/cuda/types/flags.hpp"
#include "backend/cuda/types/image2D.hpp"

namespace nova {

template <typename T>
class Image2DWrite : public Image2D<T> {
public:
  Image2DWrite() : surf(0) {}

  Image2DWrite(size_t width, size_t height) : Image2D<T>(width, height) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CUDA_CHECK_AND_THROW(
      cudaMallocArray(&this->buffer, &channel_desc, width, height, cudaArraySurfaceLoadStore))

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = this->buffer;
    CUDA_CHECK_AND_THROW(cudaCreateSurfaceObject(&surf, &res_desc))
  }

  ~Image2DWrite() { CUDA_CHECK(cudaDestroySurfaceObject(surf)) }

  Image2DWrite(Image2DWrite&& other)
    : surf(other.surf),
  Image2D<T>(std::move(other)) {
    other.surf = 0;
  }
  Image2DWrite& operator=(Image2DWrite&& other) {
    std::swap(surf, other.surf);
    Image2D<T>::operator=(std::move(other));
    return *this;
  }

  cudaSurfaceObject_t& data() { return surf; };

private:
  cudaSurfaceObject_t surf;
};

}

#endif // CUDA_IMAGE2D_WRITE_HPP