#ifndef KERNELS_BACKEND_CUDA_VECTOR_HPP
#define KERNELS_BACKEND_CUDA_VECTOR_HPP

#include <algorithm>
#include <type_traits>

#include "kernels/backend/cuda/static_if.hpp"

namespace nova {

#define NUM_COMP(u) sizeof(u) / sizeof(u.x)

template <class T>
inline constexpr bool is_arithmetic_v = std::is_arithmetic<T>::value;

template <typename W, typename U>
__device__ constexpr W xyz(U&& u) {
  return { u.x, u.y, u.z };
}

template <typename W, typename U, typename T>
__device__ constexpr W make_vector(U&& u, T t) {
  W w;
  constexpr size_t w_comp = NUM_COMP(w);
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
__device__ constexpr W make_vector(U&& u) {
  W w;

  constexpr size_t comp = NUM_COMP(w);
  static_assert(comp >= 2 && comp <= 4);
  using T = decltype(W::x);

  static_if<is_arithmetic_v<U>>([&](auto f) {
    T t = static_cast<T>(f(u));

    static_if<comp == 2>([&](auto g) {
      g(w) = { t, t };
    });
    static_if<comp == 3>([&](auto g) {
      g(w) = { t, t, t };
    });
    static_if<comp == 4>([&](auto g) {
      g(w) = { t, t, t, t };
    });
  }).else_([&](auto f) {
    static_if<comp == 2>([&](auto g) {
      g(w) = { static_cast<T>(g(u).x), static_cast<T>(g(u).y) };
    });
    static_if<comp == 3>([&](auto g) {
      g(w) = { static_cast<T>(g(u).x), static_cast<T>(g(u).y), static_cast<T>(g(u).y) };
    });
    static_if<comp == 4>([&](auto g) {
      g(w) = { static_cast<T>(g(u).x), static_cast<T>(g(u).y), static_cast<T>(g(u).z),
               static_cast<T>(g(u).w) };
    });
  });
  return w;
}

template <typename U>
__device__ constexpr auto dot(const U& u, const U& v) {
  constexpr size_t comp = NUM_COMP(u);
  static_assert(comp >= 2 && comp <= 4);

  decltype(u.x) t = 0;
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

#define VECTOR_SCALAR_BINARY_OP(op)                                    \
  template <typename A, typename B>                                    \
  __device__ constexpr auto operator op(const A& a, const B& b) {      \
    constexpr bool b_is_num = is_arithmetic_v<B>;                      \
    std::conditional_t<b_is_num, A, B> c;                              \
    static_if<b_is_num>([&](auto f) {                                  \
      constexpr size_t comp = NUM_COMP(f(a));                          \
      static_assert(comp >= 2 && comp <= 4);                           \
      static_if<comp == 2>([&](auto g) {                               \
        g(c) = { g(a).x op b, g(a).y op b };                           \
      });                                                              \
      static_if<comp == 3>([&](auto g) {                               \
        g(c) = { g(a).x op b, g(a).y op b, g(a).z op b };              \
      });                                                              \
      static_if<comp == 4>([&](auto g) {                               \
        g(c) = { g(a).x op b, g(a).y op b, g(a).z op b, g(a).w op b }; \
      });                                                              \
    }).else_([&](auto f) {                                             \
      constexpr size_t comp = NUM_COMP(f(b));                          \
      static_assert(comp >= 2 && comp <= 4);                           \
      static_if<comp == 2>([&](auto g) {                               \
        g(c) = { g(b).x op a, g(b).y op a };                           \
      });                                                              \
      static_if<comp == 3>([&](auto g) {                               \
        g(c) = { g(b).x op a, g(b).y op a, g(b).z op a };              \
      });                                                              \
      static_if<comp == 4>([&](auto g) {                               \
        g(c) = { g(b).x op a, g(b).y op a, g(b).z op a, g(b).w op a }; \
      });                                                              \
    });                                                                \
    return c;                                                          \
  }

#define VECTOR_SCALAR_FUNC(func, scalar_func)                                              \
  template <typename A, typename B>                                                        \
  __device__ constexpr auto func(const A& a, const B& b) {                                 \
    constexpr bool b_is_num = is_arithmetic_v<B>;                                          \
    std::conditional_t<b_is_num, A, B> c;                                                  \
    static_if<b_is_num>([&](auto f) {                                                      \
      constexpr size_t comp = NUM_COMP(f(a));                                              \
      static_assert(comp >= 2 && comp <= 4);                                               \
      static_if<comp == 2>([&](auto g) {                                                   \
        g(c) = { scalar_func(g(a).x, b), scalar_func(g(a).y, b) };                         \
      });                                                                                  \
      static_if<comp == 3>([&](auto g) {                                                   \
        g(c) = { scalar_func(g(a).x, b), scalar_func(g(a).y, b), scalar_func(g(a).z, b) }; \
      });                                                                                  \
      static_if<comp == 4>([&](auto g) {                                                   \
        g(c) = { scalar_func(g(a).x, b), scalar_func(g(a).y, b), scalar_func(g(a).z, b),   \
                 scalar_func(g(a).w, b) };                                                 \
      });                                                                                  \
    }).else_([&](auto f) {                                                                 \
      constexpr size_t comp = NUM_COMP(f(b));                                              \
      static_assert(comp >= 2 && comp <= 4);                                               \
      static_if<comp == 2>([&](auto g) {                                                   \
        g(c) = { scalar_func(g(b).x, a), scalar_func(g(b).y, a) };                         \
      });                                                                                  \
      static_if<comp == 3>([&](auto g) {                                                   \
        g(c) = { scalar_func(g(b).x, a), scalar_func(g(b).y, a), scalar_func(g(b).z, a) }; \
      });                                                                                  \
      static_if<comp == 4>([&](auto g) {                                                   \
        g(c) = { scalar_func(g(b).x, a), scalar_func(g(b).y, a), scalar_func(g(b).z, a),   \
                 scalar_func(g(b).w, a) };                                                 \
      });                                                                                  \
    });                                                                                    \
    return c;                                                                              \
  }

#define VECTOR_VECTOR_FUNC(func, scalar_func)                                \
  template <typename W>                                                      \
  __device__ constexpr auto func(W u, W v) {                                 \
    W w;                                                                     \
    constexpr bool is_num = is_arithmetic_v<W>;                              \
    static_if<is_num>([&](auto f) {                                          \
      f(w) = scalar_func(f(u), f(v));                                        \
    }).else_([&](auto f) {                                                   \
      constexpr size_t comp = NUM_COMP(f(u));                                \
      static_assert(comp >= 2 && comp <= 4);                                 \
      static_if<comp == 2>([&](auto g) {                                     \
        g(w) = { scalar_func(g(u).x, g(v).x), scalar_func(g(u).y, g(v).y) }; \
      });                                                                    \
      static_if<comp == 3>([&](auto g) {                                     \
        g(w) = { scalar_func(g(u).x, g(v).x), scalar_func(g(u).y, g(v).y),   \
                 scalar_func(g(u).z, g(v).z) };                              \
      });                                                                    \
      static_if<comp == 4>([&](auto g) {                                     \
        g(w) = { scalar_func(g(u).x, g(v).x), scalar_func(g(u).y, g(v).y),   \
                 scalar_func(g(u).z, g(v).z), scalar_func(g(u).w, g(v).w) }; \
      });                                                                    \
    });                                                                      \
    return w;                                                                \
  }

#define VECTOR_VECTOR_BINARY_OP(op)                                                      \
  template <typename W>                                                                  \
  __device__ constexpr W operator op(const W& u, const W& v) {                           \
    constexpr size_t comp = NUM_COMP(u);                                                 \
    static_assert(comp >= 2 && comp <= 4);                                               \
    W w;                                                                                 \
    static_if<comp == 2>([&](auto f) {                                                   \
      f(w) = { f(u).x op f(v).x, f(u).y op f(v).y };                                     \
    });                                                                                  \
    static_if<comp == 3>([&](auto f) {                                                   \
      f(w) = { f(u).x op f(v).x, f(u).y op f(v).y, f(u).z op f(v).z };                   \
    });                                                                                  \
    static_if<comp == 4>([&](auto f) {                                                   \
      f(w) = { f(u).x op f(v).x, f(u).y op f(v).y, f(u).z op f(v).z, f(u).w op f(v).w }; \
    });                                                                                  \
    return w;                                                                            \
  }

VECTOR_SCALAR_BINARY_OP(+)
VECTOR_SCALAR_BINARY_OP(-)
VECTOR_SCALAR_BINARY_OP(*)
VECTOR_SCALAR_BINARY_OP(/)

VECTOR_VECTOR_BINARY_OP(+)
VECTOR_VECTOR_BINARY_OP(-)
VECTOR_VECTOR_BINARY_OP(*)
VECTOR_VECTOR_BINARY_OP(/)

VECTOR_SCALAR_FUNC(pow, powf)

VECTOR_VECTOR_FUNC(min, fminf)
VECTOR_VECTOR_FUNC(max, fmaxf)

}

#endif