#ifndef KERNELS_BACKEND_OPENCL_VECTOR_HPP
#define KERNELS_BACKEND_OPENCL_VECTOR_HPP

#include "kernels/backend/opencl/swizzle.hpp"
#include "kernels/backend/opencl/type_traits.hpp"

namespace nova {

template <typename W, typename U, typename T>
constexpr W make_vector(U u, T t) {
  return { u, t };
}

template <typename W, typename U>
constexpr W make_vector(U u) {
  W w;
  if constexpr (is_arithmetic_v<U>) {
    w = static_cast<W>(u);
  } else {
    w = __builtin_convertvector(u, W);
  }
  return w;
}

inline float3 reflect(const float3& i, const float3& n) { return i - 2.0f * dot(n, i) * n; }

}

#endif