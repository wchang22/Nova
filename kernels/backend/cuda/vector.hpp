#ifndef KERNELS_BACKEND_CUDA_VECTOR_HPP
#define KERNELS_BACKEND_CUDA_VECTOR_HPP

#include "kernels/backend/common/static_if.hpp"

namespace nova {

template <typename W, typename U>
__device__ inline constexpr W xyz(const U& u) {
  return { u.x, u.y, u.z };
}

template <typename W, typename U, typename T>
__device__ inline constexpr W make_vector(const U& u, T t) {
  W w;
  constexpr size_t w_comp = sizeof(w) / sizeof(w.x);
  static_assert(w_comp >= 3 && w_comp <= 4);

  static_if<w_comp == 3>([&](auto f) {
    f(w) = { f(u).x, f(u).y, t };
  });
  static_if<w_comp == 4>([&](auto f) {
    f(w) = { f(u).x, f(u).y, f(u).z, t };
  });

  return w;
}

template <typename W, typename U>
__device__ inline constexpr W make_vector(const U& u) {
  constexpr size_t u_comp = sizeof(u) / sizeof(u.x);
  static_assert(u_comp >= 2 && u_comp <= 4);

  W w;
  using T = decltype(w.x);

  static_if<u_comp == 2>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y) };
  });
  static_if<u_comp == 3>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y), static_cast<T>(f(u).y) };
  });
  static_if<u_comp == 4>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y), static_cast<T>(f(u).z),
             static_cast<T>(f(u).w) };
  });
  return w;
}

template <typename U>
__device__ inline constexpr auto dot(const U& u, const U& v) {
  static_assert(sizeof(u) == sizeof(v));
  constexpr size_t u_comp = sizeof(u) / sizeof(u.x);
  constexpr size_t v_comp = sizeof(v) / sizeof(v.x);
  static_assert(sizeof(u_comp) == sizeof(v_comp) && u_comp >= 2 && u_comp <= 4);

  decltype(u.x) t = 0;
  static_if<u_comp == 2>([&](auto f) {
    f(t) = f(u).x * f(v).x + f(u).y * f(v).y;
  });
  static_if<u_comp == 3>([&](auto f) {
    f(t) = f(u).x * f(v).x + f(u).y * f(v).y + f(u).z * f(v).z;
  });
  static_if<u_comp == 4>([&](auto f) {
    f(t) = f(u).x * f(v).x + f(u).y * f(v).y + f(u).z * f(v).z + f(u).w * f(v).w;
  });
  return t;
}

#define VECTOR_SCALAR_BINARY_OP(op)                                  \
  template <typename W, typename T>                                  \
  __device__ inline constexpr W operator op(const W& u, T t) {       \
    constexpr size_t u_comp = sizeof(u) / sizeof(u.x);               \
    static_assert(u_comp >= 2 && u_comp <= 4);                       \
    W w;                                                             \
    static_if<u_comp == 2>([&](auto f) {                             \
      f(w) = { f(u).x op t, f(u).y op t };                           \
    });                                                              \
    static_if<u_comp == 3>([&](auto f) {                             \
      f(w) = { f(u).x op t, f(u).y op t, f(u).z op t };              \
    });                                                              \
    static_if<u_comp == 4>([&](auto f) {                             \
      f(w) = { f(u).x op t, f(u).y op t, f(u).z op t, f(u).w op t }; \
    });                                                              \
    return w;                                                        \
  }

#define VECTOR_SCALAR_FUNC(func)                                                   \
  template <typename W, typename T>                                                \
  __device__ inline constexpr W func(const W& u, T t) {                            \
    constexpr size_t u_comp = sizeof(u) / sizeof(u.x);                             \
    static_assert(u_comp >= 2 && u_comp <= 4);                                     \
    W w;                                                                           \
    static_if<u_comp == 2>([&](auto f) {                                           \
      f(w) = { std::func(f(u).x, t), std::func(f(u).y, t) };                       \
    });                                                                            \
    static_if<u_comp == 3>([&](auto f) {                                           \
      f(w) = { std::func(f(u).x, t), std::func(f(u).y, t), std::func(f(u).z, t) }; \
    });                                                                            \
    static_if<u_comp == 4>([&](auto f) {                                           \
      f(w) = { std::func(f(u).x, t), std::func(f(u).y, t), std::func(f(u).z, t),   \
               std::func(f(u).w, t) };                                             \
    });                                                                            \
    return w;                                                                      \
  }

#define VECTOR_VECTOR_BINARY_OP(op)                                                      \
  template <typename W>                                                                  \
  __device__ inline constexpr W operator op(const W& u, const W& v) {                    \
    constexpr size_t u_comp = sizeof(u) / sizeof(u.x);                                   \
    constexpr size_t v_comp = sizeof(v) / sizeof(v.x);                                   \
    static_assert(u_comp == v_comp && u_comp >= 2 && u_comp <= 4);                       \
    W w;                                                                                 \
    static_if<u_comp == 2>([&](auto f) {                                                 \
      f(w) = { f(u).x op f(v).x, f(u).y op f(v).y };                                     \
    });                                                                                  \
    static_if<u_comp == 3>([&](auto f) {                                                 \
      f(w) = { f(u).x op f(v).x, f(u).y op f(v).y, f(u).z op f(v).z };                   \
    });                                                                                  \
    static_if<u_comp == 4>([&](auto f) {                                                 \
      f(w) = { f(u).x op f(v).x, f(u).y op f(v).y, f(u).z op f(v).z, f(u).w op f(v).w }; \
    });                                                                                  \
    return w;                                                                            \
  }

VECTOR_SCALAR_BINARY_OP(+)
VECTOR_SCALAR_BINARY_OP(-)
VECTOR_SCALAR_BINARY_OP(*)
VECTOR_SCALAR_BINARY_OP(/)
VECTOR_SCALAR_FUNC(pow)

VECTOR_VECTOR_BINARY_OP(+)
VECTOR_VECTOR_BINARY_OP(-)
VECTOR_VECTOR_BINARY_OP(*)
VECTOR_VECTOR_BINARY_OP(/)

}

#endif