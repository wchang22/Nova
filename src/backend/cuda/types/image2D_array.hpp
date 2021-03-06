#ifndef CUDA_IMAGE2D_ARRAY_HPP
#define CUDA_IMAGE2D_ARRAY_HPP

#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "backend/cuda/types/error.hpp"
#include "backend/cuda/types/flags.hpp"

namespace nova {

template <typename T>
class Image2DArray {
public:
  Image2DArray() : tex(0), buffer(nullptr) {}

  Image2DArray(AddressMode address_mode,
               FilterMode filter_mode,
               bool normalized_coords,
               size_t array_size,
               size_t width,
               size_t height,
               std::vector<T>& data) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    cudaExtent extent = make_cudaExtent(width, height, array_size);
    CUDA_CHECK_AND_THROW(cudaMalloc3DArray(&buffer, &channel_desc, extent, cudaArrayLayered))

    cudaMemcpy3DParms copy_params;
    memset(&copy_params, 0, sizeof(copy_params));
    copy_params.srcPos = make_cudaPos(0, 0, 0);
    copy_params.dstPos = make_cudaPos(0, 0, 0);
    copy_params.srcPtr = make_cudaPitchedPtr(data.data(), width * sizeof(T), width, height);
    copy_params.dstArray = buffer;
    copy_params.extent = extent;
    copy_params.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK_AND_THROW(cudaMemcpy3D(&copy_params))

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
    CUDA_CHECK_AND_THROW(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr))
  }

  ~Image2DArray() { CUDA_CHECK(cudaDestroyTextureObject(tex)) CUDA_CHECK(cudaFreeArray(buffer)) }

  Image2DArray(Image2DArray&& other)
    : tex(other.tex),
  buffer(other.buffer) {
    other.tex = 0;
    other.buffer = 0;
  }
  Image2DArray& operator=(Image2DArray&& other) {
    std::swap(tex, other.tex);
    std::swap(buffer, other.buffer);
    return *this;
  }

  cudaTextureObject_t& data() { return tex; };

private:
  cudaTextureObject_t tex;
  cudaArray_t buffer;
};

}

#endif // CUDA_IMAGE2D_ARRAY_HPP