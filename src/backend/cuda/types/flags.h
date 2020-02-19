#ifndef CUDA_FLAGS_H
#define CUDA_FLAGS_H

enum class MemFlags {
  READ_ONLY,
  WRITE_ONLY,
  READ_WRITE,
};

enum class ImageChannelOrder {
  R,
  RGB,
  RGBA,
};

enum class ImageChannelType {
  UINT8,
  UINT32,
  INT8,
  INT32,
  FLOAT,
};

enum class AddressMode {
  WRAP = cudaAddressModeWrap,
  CLAMP = cudaAddressModeClamp,
  MIRROR = cudaAddressModeMirror,
  BORDER = cudaAddressModeBorder,
};

enum class FilterMode {
  NEAREST = cudaFilterModePoint,
  LINEAR = cudaFilterModeLinear,
};

#endif // CUDA_FLAGS_H