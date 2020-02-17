#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#include <cuda_runtime.h>

using Image2D = cl::Image2D;
using Image2DArray = cl::Image2DArray;
using Buffer = cl::Buffer;

template <typename floatn>
inline float& x(floatn& f) { return f.x; }
template <typename floatn>
inline const float& x(const floatn& f) { return f.x; }

template <typename floatn>
inline float& y(floatn& f) { return f.y; }
template <typename floatn>
inline const float& y(const floatn& f) { return f.y; }

template <typename floatn>
inline float& z(floatn& f) { return f.z; }
template <typename floatn>
inline const float& z(const floatn& f) { return f.z; }

template <typename floatn>
inline float& w(floatn& f) { return f.w; }
template <typename floatn>
inline const float& w(const floatn& f) { return f.w; }

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

#endif // CUDA_TYPES_H