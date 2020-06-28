#ifndef KERNELS_BACKEND_OPENCL_VECTOR_HPP
#define KERNELS_BACKEND_OPENCL_VECTOR_HPP

#include "kernels/backend/opencl/type_traits.hpp"

namespace nova {

template <typename V, typename U>
constexpr V xyz(U&& u) {
  return u.xyz;
}

template <typename W, typename U, typename T>
constexpr W make_vector(U&& u, T t) {
  return { u, t };
}

template <typename W, typename U>
constexpr W make_vector(U&& u) {
  W w;
  if constexpr (is_arithmetic<U>::value) {
    w = static_cast<W>(u);
  } else {
    w = __builtin_convertvector(u, W);
  }
  return w;
}

}

#endif