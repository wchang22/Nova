#include "image2D.h"
#include "backend/cuda/types/error.h"

Image2D::Image2D(AddressMode address_mode, FilterMode filter_mode, bool normalized_coords,
                 size_t width, size_t height, T* data = nullptr) {
  int pitch_in_bytes = width * sizeof(T);
  int pitch;

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

Image2D::~Image2D() {
  CUDA_CHECK(cudaDestroyTextureObject(tex))
  CUDA_CHECK(cudaFree(buffer))
}