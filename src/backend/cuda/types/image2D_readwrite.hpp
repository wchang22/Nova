#ifndef CUDA_IMAGE2D_READWRITE_HPP
#define CUDA_IMAGE2D_READWRITE_HPP

#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "backend/cuda/types/error.hpp"
#include "backend/cuda/types/flags.hpp"
#include "backend/cuda/types/image2D.hpp"

namespace nova {

template <typename T>
class Image2DReadWrite : public Image2D<T> {
public:
  Image2DReadWrite() : tex(0), surf(0) {}

  Image2DReadWrite(AddressMode address_mode,
                   FilterMode filter_mode,
                   bool normalized_coords,
                   size_t width,
                   size_t height,
                   const std::vector<T>& data = {})
    : Image2D<T>(width, height) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CUDA_CHECK_AND_THROW(cudaMallocArray(&this->buffer, &channel_desc, width, height))

    if (!data.empty()) {
      CUDA_CHECK_AND_THROW(cudaMemcpy2DToArray(this->buffer, 0, 0, data.data(), width * sizeof(T),
                                               width * sizeof(T), height, cudaMemcpyHostToDevice))
    }

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = this->buffer;
    CUDA_CHECK_AND_THROW(cudaCreateSurfaceObject(&surf, &res_desc))

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    for (int i = 0; i < 3; i++) {
      tex_desc.addressMode[i] = static_cast<cudaTextureAddressMode>(address_mode);
    }
    tex_desc.filterMode = static_cast<cudaTextureFilterMode>(filter_mode);
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = normalized_coords;
    CUDA_CHECK_AND_THROW(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr))
  }

  ~Image2DReadWrite() { CUDA_CHECK(cudaDestroyTextureObject(tex))
                          CUDA_CHECK(cudaDestroySurfaceObject(surf)) }

  Image2DReadWrite(Image2DReadWrite&& other)
    : tex(other.tex),
  surf(other.surf), Image2D<T>(std::move(other)) {
    other.tex = 0;
    other.surf = 0;
  }
  Image2DReadWrite& operator=(Image2DReadWrite&& other) {
    std::swap(tex, other.tex);
    std::swap(surf, other.surf);
    Image2D<T>::operator=(std::move(other));
    return *this;
  }

  struct ReadAccessor {
    Image2DReadWrite& image;

    const cudaTextureObject_t& data() const { return image.tex; }
  };

  struct WriteAccessor {
    Image2DReadWrite& image;

    const cudaSurfaceObject_t& data() const { return image.surf; }
  };

  const ReadAccessor read_access() const { return read_accessor; }
  const WriteAccessor write_access() const { return write_accessor; }

private:
  cudaTextureObject_t tex;
  cudaSurfaceObject_t surf;
  ReadAccessor read_accessor { *this };
  WriteAccessor write_accessor { *this };
};

}

#endif // CUDA_IMAGE2D_READWRITE_HPP