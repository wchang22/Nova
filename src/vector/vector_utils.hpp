#ifndef VECTOR_UTILS_HPP
#define VECTOR_UTILS_HPP

#include "backend/types.hpp"

namespace nova {

template <typename V, typename T, int N>
std::vector<T> flatten(const std::vector<V>& v) {
  static_assert(N >= 1 && N <= 4);

  std::vector<T> flattened_buffer;
  flattened_buffer.reserve(v.size() * N);

  for (const auto& e : v) {
    flattened_buffer.push_back(x(e));
    if constexpr (N >= 2) {
      flattened_buffer.push_back(y(e));
    }
    if constexpr (N >= 3) {
      flattened_buffer.push_back(z(e));
    }
    if constexpr (N >= 4) {
      flattened_buffer.push_back(w(e));
    }
  }

  return flattened_buffer;
}

template <typename V, typename T, int N>
std::vector<V> pack(const std::vector<T>& v) {
  static_assert(N >= 1 && N <= 4);

  std::vector<V> packed_buffer(v.size() / N);

  for (size_t i = 0; i < v.size() / N; i++) {
    x(packed_buffer[i]) = v[i * N];
    if constexpr (N >= 2) {
      y(packed_buffer[i]) = v[i * N + 1];
    }
    if constexpr (N >= 3) {
      z(packed_buffer[i]) = v[i * N + 2];
    }
    if constexpr (N >= 4) {
      w(packed_buffer[i]) = v[i * N + 3];
    }
  }

  return packed_buffer;
}

}

#endif // VECTOR_UTILS_HPP