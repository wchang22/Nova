#ifndef CUDA_IMAGE2D_ARRAY_H
#define CUDA_IMAGE2D_ARRAY_H

#include <cuda_runtime.h>
#include <utility>
#include <cstring>

#include "backend/cuda/types/error.h"
#include "backend/cuda/types/flags.h"

template<typename T>
class Image2DArray {
public:
  Image2DArray() : tex(), buffer(nullptr) {}

  Image2DArray(AddressMode address_mode, FilterMode filter_mode,
               bool normalized_coords, size_t array_size, size_t width,
               size_t height, T* data = nullptr) {
    (void) array_size;
    size_t pitch_in_bytes = width * sizeof(T);
    size_t pitch;

    CUDA_CHECK(cudaMallocPitch(&buffer, &pitch, pitch_in_bytes, height))

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = buffer;
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    res_desc.res.pitch2D.pitchInBytes = pitch;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    for (int i = 0; i < 3; i++) {
      tex_desc.addressMode[i] = static_cast<cudaTextureAddressMode>(address_mode);
    }
    tex_desc.filterMode = static_cast<cudaTextureFilterMode>(filter_mode);
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = normalized_coords;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, NULL))

    if (data) {
      CUDA_CHECK(cudaMemcpy2D(buffer, pitch, data, pitch_in_bytes, pitch_in_bytes, height,
                              cudaMemcpyHostToDevice))
    }
  }

  Image2DArray(Image2DArray&& other)
    : tex(std::move(other.tex)), buffer(std::move(other.buffer)) {
      other.buffer = nullptr;
  }
  Image2DArray& operator=(Image2DArray&& other) {
    std::swap(tex, other.tex);
    std::swap(buffer, other.buffer);
    return *this;
  }

  ~Image2DArray() {
    cudaDestroyTextureObject(tex);
    cudaFree(buffer);
  }

  cudaTextureObject_t& data() { return tex; };
  const T* ptr() const { return buffer; };

private:
  cudaTextureObject_t tex;
  T* buffer;
};

#endif // CUDA_IMAGE2D_ARRAY_H