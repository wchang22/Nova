#ifndef KERNELS_BACKEND_OPENCL_TYPE_TRAITS
#define KERNELS_BACKEND_OPENCL_TYPE_TRAITS

namespace nova {

template <typename T, T v>
struct integral_constant {
  constant static constexpr T value = v;
  using value_type = T;
  using type = integral_constant; // using injected-typename-name
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; } // since c++14
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename T>
struct remove_cv {
  typedef T type;
};
template <typename T>
struct remove_cv<const T> {
  typedef T type;
};
template <typename T>
struct remove_cv<volatile T> {
  typedef T type;
};
template <typename T>
struct remove_cv<const volatile T> {
  typedef T type;
};
template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

template <typename T>
struct remove_reference {
  typedef T type;
};
template <typename T>
struct remove_reference<T&> {
  typedef T type;
};
template <typename T>
struct remove_reference<T&&> {
  typedef T type;
};
template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

template <typename>
struct is_integral_base : false_type {};
template <typename T>
struct is_integral : is_integral_base<remove_cv_t<T>> {};
template <typename T>
constant inline constexpr bool is_integral_v = is_integral<T>::value;
template <>
struct is_integral_base<bool> : true_type {};
template <>
struct is_integral_base<int> : true_type {};
template <>
struct is_integral_base<uint> : true_type {};
template <>
struct is_integral_base<char> : true_type {};
template <>
struct is_integral_base<uchar> : true_type {};
template <>
struct is_integral_base<generic bool> : true_type {};
template <>
struct is_integral_base<generic int> : true_type {};
template <>
struct is_integral_base<generic uint> : true_type {};
template <>
struct is_integral_base<generic char> : true_type {};
template <>
struct is_integral_base<generic uchar> : true_type {};

template <typename>
struct is_floating_point_base : false_type {};
template <typename T>
struct is_floating_point : is_floating_point_base<remove_cv_t<T>> {};
template <typename T>
constant inline constexpr bool is_floating_point_v = is_floating_point<T>::value;
template <>
struct is_floating_point_base<float> : true_type {};
template <>
struct is_floating_point_base<generic float> : true_type {};

template <typename T>
struct is_arithmetic : integral_constant<bool, is_integral_v<T> || is_floating_point_v<T>> {};
template <typename T>
constant inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

namespace detail {

template <typename T, bool = is_arithmetic_v<T>>
struct is_signed : integral_constant<bool, T(-1) < T(0)> {};

template <typename T>
struct is_signed<T, false> : false_type {};

} // namespace detail

template <typename T>
struct is_signed : detail::is_signed<T>::type {};
template <typename T>
constant inline constexpr bool is_signed_v = is_signed<T>::value;

}

#endif
