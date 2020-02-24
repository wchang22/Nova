#ifndef CUDA_IMAGE2D_READ_H
#define CUDA_IMAGE2D_READ_H

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "backend/cuda/types/error.h"
#include "backend/cuda/types/flags.h"

template<typename T>
class Image2DRead {
public:
  Image2DRead(AddressMode address_mode, FilterMode filter_mode, bool normalized_coords,
              size_t width, size_t height, const std::vector<T>& data) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CUDA_CHECK(cudaMallocArray(&buffer, &channel_desc, width, height))

    CUDA_CHECK(cudaMemcpy2DToArray(buffer, 0, 0, data.data(), width * sizeof(T),    
                                   width * sizeof(T), height, cudaMemcpyHostToDevice))

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = buffer;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    for (int i = 0; i < 3; i++) {
      tex_desc.addressMode[i] = static_cast<cudaTextureAddressMode>(address_mode);
    }
    tex_desc.filterMode = static_cast<cudaTextureFilterMode>(filter_mode);
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = normalized_coords;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr))
  }

  ~Image2DRead() {
    cudaDestroyTextureObject(tex);
    cudaFreeArray(buffer);
  }

  cudaTextureObject_t& data() { return tex; };

private:
  cudaTextureObject_t tex;
  cudaArray_t buffer;
};

#endif // CUDA_IMAGE2D_READ_H