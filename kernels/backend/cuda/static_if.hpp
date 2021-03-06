#ifndef KERNELS_BACKEND_CUDA_STATIC_IF_HPP
#define KERNELS_BACKEND_CUDA_STATIC_IF_HPP

// From https://baptiste-wicht.com/posts/2015/07/simulate-static_if-with-c11c14.html

#include <utility>

namespace nova {

namespace static_if_detail {

struct identity {
  template <typename T>
  __device__ T operator()(T&& x) const {
    return std::forward<T>(x);
  }
};

template <bool Cond>
struct statement {
  template <typename F>
  __device__ void then(const F& f) {
    f(identity());
  }

  template <typename F>
  __device__ void else_(const F&) {}
};

template <>
struct statement<false> {
  template <typename F>
  __device__ void then(const F&) {}

  template <typename F>
  __device__ void else_(const F& f) {
    f(identity());
  }
};

} // end of namespace static_if_detail

template <bool Cond, typename F>
__device__ constexpr static_if_detail::statement<Cond> static_if(F const& f) {
  static_if_detail::statement<Cond> if_;
  if_.then(f);
  return if_;
}

}

#endif