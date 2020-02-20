#ifndef CUDA_IMAGE2D_H
#define CUDA_IMAGE2D_H

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "backend/cuda/types/error.h"
#include "backend/cuda/types/flags.h"

template<typename T>
class Image2D {
public:
  Image2D(AddressMode address_mode, FilterMode filter_mode, bool normalized_coords,
          size_t width, size_t height, const T* data = nullptr)
    : width(width), height(height)
  {
    size_t pitch_in_bytes = width * sizeof(T);
    size_t pitch;

    CUDA_CHECK(cudaMallocPitch(&buffer, &pitch, pitch_in_bytes, height))

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = buffer;
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    res_desc.res.pitch2D.pitchInBytes = pitch;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    for (int i = 0; i < 3; i++) {
      tex_desc.addressMode[i] = static_cast<cudaTextureAddressMode>(address_mode);
    }
    tex_desc.filterMode = static_cast<cudaTextureFilterMode>(filter_mode);
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = normalized_coords;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr))

    if (data) {
      CUDA_CHECK(cudaMemcpy2D(buffer, pitch, data, pitch_in_bytes,
                              pitch_in_bytes, height, cudaMemcpyHostToDevice))
    }
  }

  ~Image2D() {
    cudaDestroyTextureObject(tex);
    cudaFree(buffer);
  }

  cudaTextureObject_t& data() { return tex; };

  std::vector<T> read() const {
    std::vector<T> image_data(width * height);
    int pitch_in_bytes = width * sizeof(T);
    CUDA_CHECK(cudaMemcpy2D(image_data.data(), pitch_in_bytes, buffer,
                            pitch_in_bytes, pitch_in_bytes, height, cudaMemcpyDeviceToHost))
    return image_data;
  }

private:
  cudaTextureObject_t tex;
  T* buffer;
  size_t width;
  size_t height;
};

#endif // CUDA_IMAGE2D_H