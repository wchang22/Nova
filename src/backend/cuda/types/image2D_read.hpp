#ifndef CUDA_IMAGE2D_READ_HPP
#define CUDA_IMAGE2D_READ_HPP

#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "backend/cuda/types/error.hpp"
#include "backend/cuda/types/flags.hpp"
#include "backend/cuda/types/image2D.hpp"

namespace nova {

template <typename T>
class Image2DRead : public Image2D<T> {
public:
  Image2DRead(AddressMode address_mode,
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

  ~Image2DRead() { CUDA_CHECK(cudaDestroyTextureObject(tex))
                     CUDA_CHECK(cudaFreeArray(this->buffer)) }

  cudaTextureObject_t& data() {
    return tex;
  };

private:
  cudaTextureObject_t tex;
};

}

#endif // CUDA_IMAGE2D_READ_HPP