#ifndef KERNELS_BACKEND_CUDA_IMAGE_HPP
#define KERNELS_BACKEND_CUDA_IMAGE_HPP

#include "kernels/backend/common/image.hpp"

namespace nova {

using image2d_read_t = cudaTextureObject_t;
using image2d_write_t = cudaSurfaceObject_t;
using image2d_array_read_t = cudaTextureObject_t;

template <typename W, AddressMode A = AddressMode::CLAMP>
__device__ constexpr W read_image(image2d_read_t image, const int2& coords) {
  return tex2D<W>(image, coords.x, coords.y);
}

template <typename W, AddressMode A = AddressMode::CLAMP>
__device__ constexpr W read_image(image2d_read_t image, const float2& coords) {
  return tex2D<W>(image, coords.x, coords.y);
}

template <typename W, AddressMode A = AddressMode::CLAMP>
__device__ constexpr W
read_image(image2d_read_t image, const float2& coords, const float2& offset) {
  return tex2D<W>(image, coords.x + offset.x, coords.y + offset.y);
}

template <typename W, AddressMode A = AddressMode::CLAMP>
__device__ constexpr W read_image(image2d_array_read_t image, const int2& coords, int index) {
  return tex2DLayered<W>(image, coords.x, coords.y, index);
}

template <typename W, AddressMode A = AddressMode::CLAMP>
__device__ constexpr W read_image(image2d_array_read_t image, const float2& coords, int index) {
  return tex2DLayered<W>(image, coords.x, coords.y, index);
}

template <typename U>
__device__ constexpr void write_image(image2d_write_t image, const int2& coords, const U& value) {
  surf2Dwrite(value, image, coords.x * sizeof(value), coords.y);
}

}

#endif