#ifndef KERNELS_BACKEND_CUDA_SWIZZLE_HPP
#define KERNELS_BACKEND_CUDA_SWIZZLE_HPP

namespace nova {

#define CONCAT2(x, y) x##y
#define CONCAT3(x, y, z) x##y##z
#define CONCAT4(x, y, z, w) x##y##z##w

#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME

#define CONCAT(...) GET_MACRO(__VA_ARGS__, CONCAT4, CONCAT3, CONCAT2)(__VA_ARGS__)

#define ADD_SWIZZLE(...)                 \
  template <typename W, typename U>      \
  constexpr W CONCAT(__VA_ARGS__)(U u) { \
    return { u.CONCAT(__VA_ARGS__) };    \
  }

#define ADD_COMB2(...)        \
  ADD_SWIZZLE(__VA_ARGS__, x) \
  ADD_SWIZZLE(__VA_ARGS__, y) \
  ADD_SWIZZLE(__VA_ARGS__, z) \
  ADD_SWIZZLE(__VA_ARGS__, w)

#define ADD_COMB3(...)      \
  ADD_COMB2(__VA_ARGS__, x) \
  ADD_COMB2(__VA_ARGS__, y) \
  ADD_COMB2(__VA_ARGS__, z) \
  ADD_COMB2(__VA_ARGS__, w)

#define ADD_COMB4(...)      \
  ADD_COMB3(__VA_ARGS__, x) \
  ADD_COMB3(__VA_ARGS__, y) \
  ADD_COMB3(__VA_ARGS__, z) \
  ADD_COMB3(__VA_ARGS__, w)

ADD_COMB2(x)
ADD_COMB2(y)
ADD_COMB2(z)
ADD_COMB2(w)

ADD_COMB3(x)
ADD_COMB3(y)
ADD_COMB3(z)
ADD_COMB3(w)

ADD_COMB4(x)
ADD_COMB4(y)
ADD_COMB4(z)
ADD_COMB4(w)

}

#endif