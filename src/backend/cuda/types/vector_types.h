#ifndef CUDA_VECTOR_TYPES_H
#define CUDA_VECTOR_TYPES_H

#include <cuda_runtime.h>

template <typename vec_type>
inline auto& x(vec_type& f) { return f.x; }
template <typename vec_type>
inline const auto& x(const vec_type& f) { return f.x; }

template <typename vec_type>
inline auto& y(vec_type& f) { return f.y; }
template <typename vec_type>
inline const auto& y(const vec_type& f) { return f.y; }

template <typename vec_type>
inline auto& z(vec_type& f) { return f.z; }
template <typename vec_type>
inline const auto& z(const vec_type& f) { return f.z; }

template <typename vec_type>
inline auto& w(vec_type& f) { return f.w; }
template <typename vec_type>
inline const auto& w(const vec_type& f) { return f.w; }

#endif // CUDA_VECTOR_TYPES_H