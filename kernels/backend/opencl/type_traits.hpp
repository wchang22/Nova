#ifndef KERNELS_BACKEND_OPENCL_TYPE_TRAITS
#define KERNELS_BACKEND_OPENCL_TYPE_TRAITS

template <class T, T v>
struct integral_constant {
  constant static constexpr T value = v;
  using value_type = T;
  using type = integral_constant; // using injected-class-name
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; } // since c++14
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename>
struct is_integral : false_type {};
template <>
struct is_integral<bool> : true_type {};
template <>
struct is_integral<int> : true_type {};
template <>
struct is_integral<uint> : true_type {};
template <>
struct is_integral<char> : true_type {};
template <>
struct is_integral<uchar> : true_type {};

template <typename>
struct is_floating_point : false_type {};
template <>
struct is_floating_point<float> : true_type {};

template <class T>
struct is_arithmetic
  : integral_constant<bool, is_integral<T>::value || is_floating_point<T>::value> {};

#endif
