#ifndef KERNELS_BACKEND_OPENCL_VECTOR_HPP
#define KERNELS_BACKEND_OPENCL_VECTOR_HPP

#include "opencl_convert"

namespace nova {

template <typename V, typename U>
inline constexpr V xyz(const U& u) {
  return u.xyz;
}

template <typename W, typename U, typename T>
inline constexpr W make_vector(const U& u, T t) {
  return { u, t };
}

template <typename W, typename U>
inline constexpr W make_vector(const U& u) {
  return cl::convert_cast<W, U>(u);
}

}

#endif