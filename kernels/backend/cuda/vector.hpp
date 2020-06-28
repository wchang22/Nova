#ifndef KERNELS_BACKEND_CUDA_VECTOR_HPP
#define KERNELS_BACKEND_CUDA_VECTOR_HPP

#include <algorithm>

#include "kernels/backend/cuda/static_if.hpp"
#include "kernels/backend/cuda/vector_traits.hpp"

namespace nova {

template <typename W, typename U>
__device__ constexpr W xyz(U&& u) {
  static_assert(is_vector_v<U> && num_comp_v<U> >= 3);
  static_assert(is_vector_v<W> && num_comp_v<W> == 3);
  return { u.x, u.y, u.z };
}

template <typename W,
          typename U,
          typename T,
          std::enable_if_t<(is_vector_v<U> && is_arithmetic_v<T>), int> = 0>
__device__ constexpr W make_vector(U&& u, T t) {
  constexpr size_t u_comp = num_comp_v<U>;
  constexpr size_t w_comp = num_comp_v<W>;
  static_assert(u_comp == 2 || u_comp == 3);
  static_assert(is_vector_v<W> && (w_comp == u_comp + 1));

  W w;
  static_if<w_comp == 3>([&](auto f) {
    f(w) = { f(u).x, f(u).y, t };
  });
  static_if<w_comp == 4>([&](auto f) {
    f(w) = { f(u).x, f(u).y, f(u).z, t };
  });
  return w;
}

template <typename W, typename U, std::enable_if_t<(is_arithmetic_v<U> && is_vector_v<W>), int> = 0>
__device__ constexpr W make_vector(U&& u) {
  constexpr size_t comp = num_comp_v<W>;

  using T = decltype(W::x);
  W w;
  T t = static_cast<T>(u);
  static_if<comp == 2>([&](auto f) {
    f(w) = { t, t };
  });
  static_if<comp == 3>([&](auto f) {
    f(w) = { t, t, t };
  });
  static_if<comp == 4>([&](auto f) {
    f(w) = { t, t, t, t };
  });
  return w;
}

template <typename W, typename U, std::enable_if_t<(is_vector_v<U> && is_vector_v<W>), int> = 0>
__device__ constexpr W make_vector(U&& u) {
  constexpr size_t comp = num_comp_v<W>;

  using T = decltype(W::x);
  W w;
  static_if<comp == 2>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y) };
  });
  static_if<comp == 3>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y), static_cast<T>(f(u).y) };
  });
  static_if<comp == 4>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y), static_cast<T>(f(u).z),
             static_cast<T>(f(u).w) };
  });
  return w;
}

template <typename U, std::enable_if_t<is_vector_v<U>, int> = 0>
__device__ constexpr decltype(U::x) dot(const U& u, const U& v) {
  constexpr size_t comp = num_comp_v<U>;

  decltype(U::x) t = 0;
  static_if<comp == 2>([&](auto f) {
    f(t) = f(u).x * f(v).x + f(u).y * f(v).y;
  });
  static_if<comp == 3>([&](auto f) {
    f(t) = f(u).x * f(v).x + f(u).y * f(v).y + f(u).z * f(v).z;
  });
  static_if<comp == 4>([&](auto f) {
    f(t) = f(u).x * f(v).x + f(u).y * f(v).y + f(u).z * f(v).z + f(u).w * f(v).w;
  });
  return t;
}

#define VECTOR_SCALAR_FUNC(func, scalar_func)                                            \
  template <typename U, typename T,                                                      \
            std::enable_if_t<(is_vector_v<U> && is_arithmetic_v<T>), int> = 0>           \
  __device__ constexpr U func(const U& u, T t) {                                         \
    constexpr size_t comp = num_comp_v<U>;                                               \
    U w;                                                                                 \
    static_if<comp == 2>([&](auto f) {                                                   \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t) };                         \
    });                                                                                  \
    static_if<comp == 3>([&](auto f) {                                                   \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t) }; \
    });                                                                                  \
    static_if<comp == 4>([&](auto f) {                                                   \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t),   \
               scalar_func(f(u).w, t) };                                                 \
    });                                                                                  \
    return w;                                                                            \
  }                                                                                      \
  template <typename T, typename U,                                                      \
            std::enable_if_t<(is_vector_v<U> && is_arithmetic_v<T>), int> = 0>           \
  __device__ constexpr U func(T t, const U& u) {                                         \
    constexpr size_t comp = num_comp_v<U>;                                               \
    U w;                                                                                 \
    static_if<comp == 2>([&](auto f) {                                                   \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t) };                         \
    });                                                                                  \
    static_if<comp == 3>([&](auto f) {                                                   \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t) }; \
    });                                                                                  \
    static_if<comp == 4>([&](auto f) {                                                   \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t),   \
               scalar_func(f(u).w, t) };                                                 \
    });                                                                                  \
    return w;                                                                            \
  }

#define VECTOR_VECTOR_FUNC(func, scalar_func)                              \
  template <typename T, std::enable_if_t<is_arithmetic_v<T>, int> = 0>     \
  __device__ constexpr T func(T s, T t) {                                  \
    return scalar_func(s, t);                                              \
  }                                                                        \
  template <typename U, std::enable_if_t<is_vector_v<U>, int> = 0>         \
  __device__ constexpr U func(const U& u, const U& v) {                    \
    U w;                                                                   \
    constexpr size_t comp = num_comp_v<U>;                                 \
    static_if<comp == 2>([&](auto f) {                                     \
      f(w) = { scalar_func(f(u).x, f(v).x), scalar_func(f(u).y, f(v).y) }; \
    });                                                                    \
    static_if<comp == 3>([&](auto f) {                                     \
      f(w) = { scalar_func(f(u).x, f(v).x), scalar_func(f(u).y, f(v).y),   \
               scalar_func(f(u).z, f(v).z) };                              \
    });                                                                    \
    static_if<comp == 4>([&](auto f) {                                     \
      f(w) = { scalar_func(f(u).x, f(v).x), scalar_func(f(u).y, f(v).y),   \
               scalar_func(f(u).z, f(v).z), scalar_func(f(u).w, f(v).w) }; \
    });                                                                    \
    return w;                                                              \
  }

#define VECTOR_UNARY_FUNC(func, scalar_func)                                    \
  template <typename T, std::enable_if_t<is_arithmetic_v<T>, int> = 0>          \
  __device__ constexpr T func(T t) {                                            \
    return scalar_func(t);                                                      \
  }                                                                             \
  template <typename U, std::enable_if_t<is_vector_v<U>, int> = 0>              \
  __device__ constexpr U func(const U& u) {                                     \
    U w;                                                                        \
    constexpr size_t comp = num_comp_v<U>;                                      \
    static_if<comp == 2>([&](auto f) {                                          \
      f(w) = { scalar_func(f(u).x), scalar_func(f(u).y) };                      \
    });                                                                         \
    static_if<comp == 3>([&](auto f) {                                          \
      f(w) = { scalar_func(f(u).x), scalar_func(f(u).y), scalar_func(f(u).z) }; \
    });                                                                         \
    static_if<comp == 4>([&](auto f) {                                          \
      f(w) = { scalar_func(f(u).x), scalar_func(f(u).y), scalar_func(f(u).z),   \
               scalar_func(f(u).w) };                                           \
    });                                                                         \
    return w;                                                                   \
  }

#define op_add         \
  [](auto a, auto b) { \
    return a + b;      \
  }
#define op_sub         \
  [](auto a, auto b) { \
    return a - b;      \
  }
#define op_mul         \
  [](auto a, auto b) { \
    return a * b;      \
  }
#define op_div         \
  [](auto a, auto b) { \
    return a / b;      \
  }
#define op_neg \
  [](auto a) { \
    return -a; \
  }

VECTOR_SCALAR_FUNC(operator+, op_add)
VECTOR_SCALAR_FUNC(operator-, op_sub)
VECTOR_SCALAR_FUNC(operator*, op_mul)
VECTOR_SCALAR_FUNC(operator/, op_div)
VECTOR_SCALAR_FUNC(pow, powf)

VECTOR_VECTOR_FUNC(operator+, op_add)
VECTOR_VECTOR_FUNC(operator-, op_sub)
VECTOR_VECTOR_FUNC(operator*, op_mul)
VECTOR_VECTOR_FUNC(operator/, op_div)
VECTOR_VECTOR_FUNC(min, fminf)
VECTOR_VECTOR_FUNC(max, fmaxf)

VECTOR_UNARY_FUNC(operator-, op_neg)
VECTOR_UNARY_FUNC(exp, expf)

}

#endif