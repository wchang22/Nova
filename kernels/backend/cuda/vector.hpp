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

#define VECTOR_SCALAR_FUNC(func, scalar_func, vector_type, scalar_type)                      \
  template <typename U, typename T,                                                          \
            std::enable_if_t<(is_##vector_type##_v<U> && is_##scalar_type##_v<T>), int> = 0> \
  __device__ constexpr U func(const U& u, T t) {                                             \
    constexpr size_t comp = num_comp_v<U>;                                                   \
    U w;                                                                                     \
    static_if<comp == 2>([&](auto f) {                                                       \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t) };                             \
    });                                                                                      \
    static_if<comp == 3>([&](auto f) {                                                       \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t) };     \
    });                                                                                      \
    static_if<comp == 4>([&](auto f) {                                                       \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t),       \
               scalar_func(f(u).w, t) };                                                     \
    });                                                                                      \
    return w;                                                                                \
  }                                                                                          \
  template <typename T, typename U,                                                          \
            std::enable_if_t<(is_##vector_type##_v<U> && is_##scalar_type##_v<T>), int> = 0> \
  __device__ constexpr U func(T t, const U& u) {                                             \
    constexpr size_t comp = num_comp_v<U>;                                                   \
    U w;                                                                                     \
    static_if<comp == 2>([&](auto f) {                                                       \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t) };                             \
    });                                                                                      \
    static_if<comp == 3>([&](auto f) {                                                       \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t) };     \
    });                                                                                      \
    static_if<comp == 4>([&](auto f) {                                                       \
      f(w) = { scalar_func(f(u).x, t), scalar_func(f(u).y, t), scalar_func(f(u).z, t),       \
               scalar_func(f(u).w, t) };                                                     \
    });                                                                                      \
    return w;                                                                                \
  }

#define VECTOR_VECTOR_FUNC(func, scalar_func, vector_type, scalar_type)     \
  template <typename T, std::enable_if_t<is_##scalar_type##_v<T>, int> = 0> \
  __device__ constexpr T func(T s, T t) {                                   \
    return scalar_func(s, t);                                               \
  }                                                                         \
  template <typename U, std::enable_if_t<is_##vector_type##_v<U>, int> = 0> \
  __device__ constexpr U func(const U& u, const U& v) {                     \
    U w;                                                                    \
    constexpr size_t comp = num_comp_v<U>;                                  \
    static_if<comp == 2>([&](auto f) {                                      \
      f(w) = { scalar_func(f(u).x, f(v).x), scalar_func(f(u).y, f(v).y) };  \
    });                                                                     \
    static_if<comp == 3>([&](auto f) {                                      \
      f(w) = { scalar_func(f(u).x, f(v).x), scalar_func(f(u).y, f(v).y),    \
               scalar_func(f(u).z, f(v).z) };                               \
    });                                                                     \
    static_if<comp == 4>([&](auto f) {                                      \
      f(w) = { scalar_func(f(u).x, f(v).x), scalar_func(f(u).y, f(v).y),    \
               scalar_func(f(u).z, f(v).z), scalar_func(f(u).w, f(v).w) };  \
    });                                                                     \
    return w;                                                               \
  }

#define VECTOR_UNARY_FUNC(func, scalar_func, vector_type, scalar_type)          \
  template <typename T, std::enable_if_t<is_##scalar_type##_v<T>, int> = 0>     \
  __device__ constexpr T func(T t) {                                            \
    return scalar_func(t);                                                      \
  }                                                                             \
  template <typename U, std::enable_if_t<is_##vector_type##_v<U>, int> = 0>     \
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

#define VECTOR_COMP_FUNC(func, scalar_func)                                                \
  template <typename T, std::enable_if_t<is_arithmetic_v<T>, int> = 0>                     \
  __device__ constexpr int func(T s, T t) {                                                \
    return scalar_func(s, t);                                                              \
  }                                                                                        \
  template <typename U, std::enable_if_t<(is_vector_v<U> && num_comp_v<U> == 2), int> = 0> \
  __device__ constexpr int2 func(const U& u, const U& v) {                                 \
    return { scalar_func(u.x, v.x), scalar_func(u.y, v.y) };                               \
  }                                                                                        \
  template <typename U, std::enable_if_t<(is_vector_v<U> && num_comp_v<U> == 3), int> = 0> \
  __device__ constexpr int3 func(const U& u, const U& v) {                                 \
    return { scalar_func(u.x, v.x), scalar_func(u.y, v.y), scalar_func(u.z, v.z) };        \
  }                                                                                        \
  template <typename U, std::enable_if_t<(is_vector_v<U> && num_comp_v<U> == 4), int> = 0> \
  __device__ constexpr int4 func(const U& u, const U& v) {                                 \
    return { scalar_func(u.x, v.x), scalar_func(u.y, v.y), scalar_func(u.z, v.z),          \
             scalar_func(u.w, v.w) };                                                      \
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
#define op_less        \
  [](auto a, auto b) { \
    return a < b;      \
  }
#define op_lessequal   \
  [](auto a, auto b) { \
    return a <= b;     \
  }
#define op_greater     \
  [](auto a, auto b) { \
    return a > b;      \
  }
#define op_greaterequal \
  [](auto a, auto b) {  \
    return a >= b;      \
  }
#define op_equal       \
  [](auto a, auto b) { \
    return a == b;     \
  }
#define op_notequal    \
  [](auto a, auto b) { \
    return a != b;     \
  }

VECTOR_SCALAR_FUNC(operator+, op_add, vector, arithmetic)
VECTOR_SCALAR_FUNC(operator-, op_sub, vector, arithmetic)
VECTOR_SCALAR_FUNC(operator*, op_mul, vector, arithmetic)
VECTOR_SCALAR_FUNC(operator/, op_div, vector, arithmetic)
VECTOR_SCALAR_FUNC(pow, powf, float_vector, float)

VECTOR_VECTOR_FUNC(operator+, op_add, vector, arithmetic)
VECTOR_VECTOR_FUNC(operator-, op_sub, vector, arithmetic)
VECTOR_VECTOR_FUNC(operator*, op_mul, vector, arithmetic)
VECTOR_VECTOR_FUNC(operator/, op_div, vector, arithmetic)
VECTOR_VECTOR_FUNC(min, fminf, float_vector, float)
VECTOR_VECTOR_FUNC(max, fmaxf, float_vector, float)
VECTOR_VECTOR_FUNC(min, min, integral_vector, integral)
VECTOR_VECTOR_FUNC(max, max, integral_vector, integral)
VECTOR_VECTOR_FUNC(fmod, fmodf, float_vector, float)
VECTOR_VECTOR_FUNC(floor, floorf, float_vector, float)

VECTOR_UNARY_FUNC(operator-, op_neg, vector, arithmetic)
VECTOR_UNARY_FUNC(exp, expf, float_vector, float)
VECTOR_UNARY_FUNC(fabs, fabs, float_vector, float)

VECTOR_COMP_FUNC(isless, op_less)
VECTOR_COMP_FUNC(islessequal, op_lessequal)
VECTOR_COMP_FUNC(isgreater, op_greater)
VECTOR_COMP_FUNC(isgreaterequal, op_greaterequal)
VECTOR_COMP_FUNC(isequal, op_equal)
VECTOR_COMP_FUNC(isnotequal, op_notequal)

template <typename U, std::enable_if_t<is_float_vector_v<U>, int> = 0>
__device__ constexpr decltype(U::x) dot(const U& u, const U& v) {
  constexpr size_t comp = num_comp_v<U>;

  decltype(U::x) t;
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

template <typename U, std::enable_if_t<is_float_vector_v<U>, int> = 0>
__device__ constexpr float length(const U& u) {
  return sqrtf(dot(u, u));
}

template <typename U, std::enable_if_t<is_float_vector_v<U>, int> = 0>
__device__ constexpr float distance(const U& u, const U& v) {
  return length(u - v);
}

template <typename U, std::enable_if_t<is_float_vector_v<U>, int> = 0>
__device__ constexpr U mix(const U& u, const U& v, float alpha) {
  return (1.0f - alpha) * u + alpha * v;
}

template <typename U, std::enable_if_t<is_float_vector_v<U>, int> = 0>
__device__ constexpr U normalize(const U& u) {
  return u * rsqrtf(dot(u, u));
}

template <typename T, std::enable_if_t<is_arithmetic_v<T>, int> = 0>
__device__ constexpr T clamp(T t, T lo, T hi) {
  return max(hi, min(t, lo));
}

template <typename U, typename T, std::enable_if_t<(is_vector_v<U> && is_arithmetic_v<T>), int> = 0>
__device__ constexpr U clamp(const U& u, T lo, T hi) {
  constexpr size_t comp = num_comp_v<U>;

  U w;
  static_if<comp == 2>([&](auto f) {
    f(w) = { clamp(f(u).x, lo, hi), clamp(f(u).y, lo, hi) };
  });
  static_if<comp == 3>([&](auto f) {
    f(w) = { clamp(f(u).x, lo, hi), clamp(f(u).y, lo, hi), clamp(f(u).z, lo, hi) };
  });
  static_if<comp == 4>([&](auto f) {
    f(w) = { clamp(f(u).x, lo, hi), clamp(f(u).y, lo, hi), clamp(f(u).z, lo, hi),
             clamp(f(u).w, lo, hi) };
  });
  return w;
}

template <typename U, std::enable_if_t<is_vector_v<U>, int> = 0>
__device__ constexpr decltype(U::x) clamp(const U& u, const U& lo, const U& hi) {
  constexpr size_t comp = num_comp_v<U>;

  U w;
  static_if<comp == 2>([&](auto f) {
    f(w) = { clamp(f(u).x, f(lo).x, f(hi).x), clamp(f(u).y, f(lo).y, f(hi).y) };
  });
  static_if<comp == 3>([&](auto f) {
    f(w) = { clamp(f(u).x, f(lo).x, f(hi).x), clamp(f(u).y, f(lo).y, f(hi).y),
             clamp(f(u).z, f(lo).z, f(hi).z) };
  });
  static_if<comp == 4>([&](auto f) {
    f(w) = { clamp(f(u).x, f(lo).x, f(hi).x), clamp(f(u).y, f(lo).y, f(hi).y),
             clamp(f(u).z, f(lo).z, f(hi).z), clamp(f(u).w, f(lo).w, f(hi).w) };
  });
  return w;
}

template <typename T, std::enable_if_t<is_integral_v<T>, int> = 0>
__device__ constexpr int all(T t) {
  return static_cast<int>(t);
}
template <typename U, std::enable_if_t<(is_integral_vector_v<U> && num_comp_v<U> == 2), int> = 0>
__device__ constexpr int all(const U& u) {
  return u.x && u.y;
}
template <typename U, std::enable_if_t<(is_integral_vector_v<U> && num_comp_v<U> == 3), int> = 0>
__device__ constexpr int all(const U& u) {
  return u.x && u.y && u.z;
}
template <typename U, std::enable_if_t<(is_integral_vector_v<U> && num_comp_v<U> == 4), int> = 0>
__device__ constexpr int all(const U& u) {
  return u.x && u.y && u.z && u.w;
}

template <typename T, std::enable_if_t<is_integral_v<T>, int> = 0>
__device__ constexpr int any(T t) {
  return static_cast<int>(t);
}
template <typename U, std::enable_if_t<(is_integral_vector_v<U> && num_comp_v<U> == 2), int> = 0>
__device__ constexpr int any(const U& u) {
  return u.x || u.y;
}
template <typename U, std::enable_if_t<(is_integral_vector_v<U> && num_comp_v<U> == 3), int> = 0>
__device__ constexpr int any(const U& u) {
  return u.x || u.y || u.z;
}
template <typename U, std::enable_if_t<(is_integral_vector_v<U> && num_comp_v<U> == 4), int> = 0>
__device__ constexpr int any(const U& u) {
  return u.x || u.y || u.z || u.w;
}

}

#endif