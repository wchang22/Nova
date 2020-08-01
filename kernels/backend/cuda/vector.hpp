#ifndef KERNELS_BACKEND_CUDA_VECTOR_HPP
#define KERNELS_BACKEND_CUDA_VECTOR_HPP

#include "kernels/backend/cuda/static_if.hpp"
#include "kernels/backend/cuda/swizzle.hpp"
#include "kernels/backend/cuda/vector_traits.hpp"

namespace nova {

template <typename W,
          typename U,
          typename T,
          std::enable_if_t<(is_vector_v<U> && is_arithmetic_v<T>), int> = 0>
__device__ constexpr W make_vector(U&& u, T t) {
  constexpr size_t w_comp = num_comp_v<W>;

  W w {};
  static_if<w_comp == 3>([&](auto f) {
    f(w) = { f(u).x, f(u).y, t };
  });
  static_if<w_comp == 4>([&](auto f) {
    f(w) = { f(u).x, f(u).y, f(u).z, t };
  });
  return w;
}

template <typename W, typename T, std::enable_if_t<(is_arithmetic_v<T> && is_vector_v<W>), int> = 0>
__device__ constexpr W make_vector(T t) {
  constexpr size_t comp = num_comp_v<W>;

  using S = decltype(W::x);
  W w {};
  S s = static_cast<S>(t);
  static_if<comp == 2>([&](auto f) {
    f(w) = { s, s };
  });
  static_if<comp == 3>([&](auto f) {
    f(w) = { s, s, s };
  });
  static_if<comp == 4>([&](auto f) {
    f(w) = { s, s, s, s };
  });
  return w;
}

template <typename W, typename U, std::enable_if_t<(is_vector_v<U> && is_vector_v<W>), int> = 0>
__device__ constexpr W make_vector(U&& u) {
  constexpr size_t comp = num_comp_v<W>;

  using T = decltype(W::x);
  W w {};
  static_if<comp == 2>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y) };
  });
  static_if<comp == 3>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y), static_cast<T>(f(u).z) };
  });
  static_if<comp == 4>([&](auto f) {
    f(w) = { static_cast<T>(f(u).x), static_cast<T>(f(u).y), static_cast<T>(f(u).z),
             static_cast<T>(f(u).w) };
  });
  return w;
}

#define VECTOR_SCALAR_OPERATOR(func, scalar_func, vector_type, scalar_type)                  \
  template <typename U, typename T,                                                          \
            std::enable_if_t<(is_##vector_type##_v<U> && is_##scalar_type##_v<T>), int> = 0> \
  __device__ constexpr U func(const U& u, T t) {                                             \
    constexpr size_t comp = num_comp_v<U>;                                                   \
    U w {};                                                                                  \
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
    U w {};                                                                                  \
    static_if<comp == 2>([&](auto f) {                                                       \
      f(w) = { scalar_func(t, f(u).x), scalar_func(t, f(u).y) };                             \
    });                                                                                      \
    static_if<comp == 3>([&](auto f) {                                                       \
      f(w) = { scalar_func(t, f(u).x), scalar_func(t, f(u).y), scalar_func(t, f(u).z) };     \
    });                                                                                      \
    static_if<comp == 4>([&](auto f) {                                                       \
      f(w) = { scalar_func(t, f(u).x), scalar_func(t, f(u).y), scalar_func(t, f(u).z),       \
               scalar_func(t, f(u).w) };                                                     \
    });                                                                                      \
    return w;                                                                                \
  }

#define VECTOR_SCALAR_FUNC(func, scalar_func, vector_type, scalar_type) \
  VECTOR_SCALAR_OPERATOR(func, scalar_func, vector_type, scalar_type)   \
  __device__ constexpr scalar_type func(scalar_type s, scalar_type t) { return scalar_func(s, t); }

#define VECTOR_VECTOR_FUNC(func, scalar_func, vector_type, scalar_type)     \
  template <typename T, std::enable_if_t<is_##scalar_type##_v<T>, int> = 0> \
  __device__ constexpr T func(T s, T t) {                                   \
    return scalar_func(s, t);                                               \
  }                                                                         \
  template <typename U, std::enable_if_t<is_##vector_type##_v<U>, int> = 0> \
  __device__ constexpr U func(const U& u, const U& v) {                     \
    U w {};                                                                 \
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
    U w {};                                                                     \
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

#define VECTOR_ASSIGNMENT(op)                                                                      \
  template <typename U, typename T,                                                                \
            std::enable_if_t<(is_vector_v<U> && is_arithmetic_v<T>), int> = 0>                     \
  __device__ constexpr U& operator op(U& u, T t) {                                                 \
    constexpr size_t comp = num_comp_v<U>;                                                         \
    static_if<comp == 2>([&](auto f) {                                                             \
      f(u).x op t;                                                                                 \
      f(u).y op t;                                                                                 \
    });                                                                                            \
    static_if<comp == 3>([&](auto f) {                                                             \
      f(u).x op t;                                                                                 \
      f(u).y op t;                                                                                 \
      f(u).z op t;                                                                                 \
    });                                                                                            \
    static_if<comp == 4>([&](auto f) {                                                             \
      f(u).x op t;                                                                                 \
      f(u).y op t;                                                                                 \
      f(u).z op t;                                                                                 \
      f(u).w op t;                                                                                 \
    });                                                                                            \
    return u;                                                                                      \
  }                                                                                                \
  template <typename U, typename V,                                                                \
            std::enable_if_t<(is_vector_v<U> && is_vector_v<V> && num_comp_v<U> == num_comp_v<V>), \
                             int> = 0>                                                             \
  __device__ constexpr U& operator op(U& u, const V& v) {                                          \
    constexpr size_t comp = num_comp_v<U>;                                                         \
    static_if<comp == 2>([&](auto f) {                                                             \
      f(u).x op f(v).x;                                                                            \
      f(u).y op f(v).y;                                                                            \
    });                                                                                            \
    static_if<comp == 3>([&](auto f) {                                                             \
      f(u).x op f(v).x;                                                                            \
      f(u).y op f(v).y;                                                                            \
      f(u).z op f(v).z;                                                                            \
    });                                                                                            \
    static_if<comp == 4>([&](auto f) {                                                             \
      f(u).x op f(v).x;                                                                            \
      f(u).y op f(v).y;                                                                            \
      f(u).z op f(v).z;                                                                            \
      f(u).w op f(v).w;                                                                            \
    });                                                                                            \
    return u;                                                                                      \
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
#define op_min            \
  [](auto a, auto b) {    \
    return a < b ? a : b; \
  }
#define op_max            \
  [](auto a, auto b) {    \
    return a > b ? a : b; \
  }

VECTOR_SCALAR_OPERATOR(operator+, op_add, vector, arithmetic)
VECTOR_SCALAR_OPERATOR(operator-, op_sub, vector, arithmetic)
VECTOR_SCALAR_OPERATOR(operator*, op_mul, vector, arithmetic)
VECTOR_SCALAR_OPERATOR(operator/, op_div, vector, arithmetic)
VECTOR_SCALAR_FUNC(pow, powf, float_vector, float)

VECTOR_VECTOR_FUNC(operator+, op_add, vector, arithmetic)
VECTOR_VECTOR_FUNC(operator-, op_sub, vector, arithmetic)
VECTOR_VECTOR_FUNC(operator*, op_mul, vector, arithmetic)
VECTOR_VECTOR_FUNC(operator/, op_div, vector, arithmetic)
VECTOR_VECTOR_FUNC(min, fminf, float_vector, float)
VECTOR_VECTOR_FUNC(max, fmaxf, float_vector, float)
VECTOR_VECTOR_FUNC(min, op_min, integral_vector, integral)
VECTOR_VECTOR_FUNC(max, op_max, integral_vector, integral)
VECTOR_VECTOR_FUNC(fmod, fmodf, float_vector, float)
VECTOR_VECTOR_FUNC(floor, floorf, float_vector, float)

VECTOR_UNARY_FUNC(operator-, op_neg, vector, arithmetic)
VECTOR_UNARY_FUNC(exp, expf, float_vector, float)
VECTOR_UNARY_FUNC(fabs, fabsf, float_vector, float)

VECTOR_COMP_FUNC(isless, op_less)
VECTOR_COMP_FUNC(islessequal, op_lessequal)
VECTOR_COMP_FUNC(isgreater, op_greater)
VECTOR_COMP_FUNC(isgreaterequal, op_greaterequal)
VECTOR_COMP_FUNC(isequal, op_equal)
VECTOR_COMP_FUNC(isnotequal, op_notequal)

VECTOR_ASSIGNMENT(+=)
VECTOR_ASSIGNMENT(-=)
VECTOR_ASSIGNMENT(*=)
VECTOR_ASSIGNMENT(/=)

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
  return max(lo, min(t, hi));
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

__device__ constexpr float3 reflect(const float3& i, const float3& n) {
  return i - 2.0f * dot(n, i) * n;
}

__device__ constexpr float3 cross(const float3& a, const float3& b) {
  return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
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