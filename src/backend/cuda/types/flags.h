#ifndef CUDA_FLAGS_H
#define CUDA_FLAGS_H

enum class MemFlags {
  READ_ONLY = 0,
  WRITE_ONLY = 1,
  READ_WRITE = 2,
};

enum class ImageChannelOrder {
  RGB = 0,
  RGBA = 1,
  R = 2,
};

enum class ImageChannelType {
  UINT8 = 0,
  UINT32 = 1,
  INT8 = 2,
  INT32 = 3,
  FLOAT = 4,
};

#endif // CUDA_FLAGS_H