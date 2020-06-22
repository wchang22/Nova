#ifndef KERNELS_BACKEND_COMMON_VECTOR_HPP
#define KERNELS_BACKEND_COMMON_VECTOR_HPP

#include "kernels/backend/kernel.hpp"

namespace nova {

template <typename W, typename T>
DEVICE inline constexpr W make_vector(T a, T b) {
  return { a, b };
}

template <typename W, typename T>
DEVICE inline constexpr W make_vector(T a, T b, T c) {
  return { a, b, c };
}

template <typename W, typename T>
DEVICE inline constexpr W make_vector(T a, T b, T c, T d) {
  return { a, b, c, d };
}

}

#endif