#ifndef CUDA_VECTOR_TYPES_H
#define CUDA_VECTOR_TYPES_H

#include <cuda_runtime.h>

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

#endif // CUDA_VECTOR_TYPES_H