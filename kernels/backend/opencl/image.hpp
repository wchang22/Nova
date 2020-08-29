#ifndef KERNELS_BACKEND_OPENCL_IMAGE_HPP
#define KERNELS_BACKEND_OPENCL_IMAGE_HPP

#include "kernels/backend/common/image.hpp"
#include "kernels/backend/opencl/type_traits.hpp"
#include "kernels/backend/vector.hpp"

namespace nova {

#define image2d_read_t read_only image2d_t
#define image2d_write_t write_only image2d_t
#define image2d_array_read_t read_only image2d_array_t
#define image2d_array_write_t write_only image2d_array_t

constant sampler_t sampler_int_clamp =
  CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
constant sampler_t sampler_int_wrap =
  CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
constant sampler_t sampler_float_clamp =
  CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;
constant sampler_t sampler_float_wrap =
  CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;

template <typename W, AddressMode A = AddressMode::CLAMP>
inline W read_image(image2d_read_t image, const int2& coords) {
  W w {};
  using T = remove_reference_t<decltype(w.x)>;
  if constexpr (A == AddressMode::WRAP) {
    if constexpr (is_signed_v<T>) {
      w = make_vector<W>(read_imagei(image, sampler_int_wrap, coords));
    } else {
      w = make_vector<W>(read_imageui(image, sampler_int_wrap, coords));
    }
  } else {
    if constexpr (is_signed_v<T>) {
      w = make_vector<W>(read_imagei(image, sampler_int_clamp, coords));
    } else {
      w = make_vector<W>(read_imageui(image, sampler_int_clamp, coords));
    }
  }
  return w;
}

template <typename W, AddressMode A = AddressMode::CLAMP>
inline W read_image(image2d_read_t image, const float2& coords) {
  W w {};
  if constexpr (A == AddressMode::WRAP) {
    w = make_vector<W>(read_imagef(image, sampler_float_wrap, coords));
  } else {
    w = make_vector<W>(read_imagef(image, sampler_float_clamp, coords));
  }
  return w;
}

template <typename W, AddressMode A = AddressMode::CLAMP>
inline W read_image(image2d_read_t image, const float2& coords, const float2& offset) {
  W w {};
  if constexpr (A == AddressMode::WRAP) {
    w = make_vector<W>(read_imagef(image, sampler_float_wrap, coords + offset));
  } else {
    w = make_vector<W>(read_imagef(image, sampler_float_clamp, coords + offset));
  }
  return w;
}

template <typename W, AddressMode A = AddressMode::CLAMP>
inline W read_image(image2d_array_read_t image, const int2& coords, int index) {
  W w {};
  using T = remove_reference_t<decltype(w.x)>;
  if constexpr (A == AddressMode::WRAP) {
    if constexpr (is_signed_v<T>) {
      w = make_vector<W>(read_imagei(image, sampler_int_wrap, int4 { coords, index, 0 }));
    } else {
      w = make_vector<W>(read_imageui(image, sampler_int_wrap, int4 { coords, index, 0 }));
    }
  } else {
    if constexpr (is_signed_v<T>) {
      w = make_vector<W>(read_imagei(image, sampler_int_clamp, int4 { coords, index, 0 }));
    } else {
      w = make_vector<W>(read_imageui(image, sampler_int_clamp, int4 { coords, index, 0 }));
    }
  }
  return w;
}

template <typename W, AddressMode A = AddressMode::CLAMP>
inline W read_image(image2d_array_read_t image, const float2& coords, int index) {
  W w {};
  if constexpr (A == AddressMode::WRAP) {
    w = make_vector<W>(
      read_imagef(image, sampler_float_wrap, float4 { coords, static_cast<float>(index), 0.0f }));
  } else {
    w = make_vector<W>(
      read_imagef(image, sampler_float_clamp, float4 { coords, static_cast<float>(index), 0.0f }));
  }
  return w;
}


template <typename W, AddressMode A = AddressMode::CLAMP>
inline W read_image(image2d_array_read_t image, const float2& coords, int index, const float2& offset) {
  W w {};
  if constexpr (A == AddressMode::WRAP) {
    w = make_vector<W>(
      read_imagef(image, sampler_float_wrap, float4 { coords + offset, static_cast<float>(index), 0.0f }));
  } else {
    w = make_vector<W>(
      read_imagef(image, sampler_float_clamp, float4 { coords + offset, static_cast<float>(index), 0.0f }));
  }
  return w;
}

template <typename U>
inline void write_image(image2d_write_t image, const int2& coords, const U& value) {
  using T = remove_reference_t<decltype(value.x)>;
  if constexpr (is_floating_point_v<T>) {
    write_imagef(image, coords, value);
  } else if constexpr (is_signed_v<T>) {
    write_imagei(image, coords, make_vector<int4>(value));
  } else {
    write_imageui(image, coords, make_vector<uint4>(value));
  }
}

template <typename U>
inline void write_image(image2d_array_write_t image, const int2& coords, int index, const U& value) {
  using T = remove_reference_t<decltype(value.x)>;
  if constexpr (is_floating_point_v<T>) {
    write_imagef(image, int4 { coords, index, 0 }, value);
  } else if constexpr (is_signed_v<T>) {
    write_imagei(image, int4 { coords, index, 0 }, make_vector<int4>(value));
  } else {
    write_imageui(image, int4 { coords, index, 0 }, make_vector<uint4>(value));
  }
}

}

#endif