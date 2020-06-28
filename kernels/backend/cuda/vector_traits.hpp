#ifndef KERNELS_BACKEND_CUDA_VECTOR_TRAITS_HPP
#define KERNELS_BACKEND_CUDA_VECTOR_TRAITS_HPP

#include <type_traits>

namespace nova {

#define NUM_COMP(v) sizeof(v) / sizeof(v.x)

template <typename T>
inline constexpr bool is_arithmetic_v = std::is_arithmetic<T>::value;

template <typename T>
struct num_comp_base : std::integral_constant<size_t, 0> {};
template <typename T>
struct num_comp : num_comp_base<std::decay_t<T>> {};
template <typename T>
__device__ inline constexpr size_t num_comp_v = num_comp<T>::value;

template <typename T>
struct is_vector_base : std::false_type {};
template <typename T>
struct is_vector : is_vector_base<std::decay_t<T>> {};
template <class T>
__device__ inline constexpr bool is_vector_v = is_vector<T>::value;

#define ADD_VECTOR_TYPE(scalar)                                           \
  template <>                                                             \
  struct is_vector_base<scalar##2> : std::true_type {};                   \
  template <>                                                             \
  struct is_vector_base<scalar##3> : std::true_type {};                   \
  template <>                                                             \
  struct is_vector_base<scalar##4> : std::true_type {};                   \
  template <>                                                             \
  struct num_comp_base<scalar##2> : std::integral_constant<size_t, 2> {}; \
  template <>                                                             \
  struct num_comp_base<scalar##3> : std::integral_constant<size_t, 3> {}; \
  template <>                                                             \
  struct num_comp_base<scalar##4> : std::integral_constant<size_t, 4> {};

ADD_VECTOR_TYPE(float)
ADD_VECTOR_TYPE(int)
ADD_VECTOR_TYPE(uint)
ADD_VECTOR_TYPE(char)
ADD_VECTOR_TYPE(uchar)
ADD_VECTOR_TYPE(short)
ADD_VECTOR_TYPE(ushort)

}

#endif