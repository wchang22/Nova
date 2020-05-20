#ifndef IMAGEUTILS_HPP
#define IMAGEUTILS_HPP

#include <cstdint>
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>
#include <vector>

#include "backend/types.hpp"
#include "util/exception/exception.hpp"

namespace image_utils {

template <typename vec_type>
struct image {
  std::vector<vec_type> data;
  uint32_t width;
  uint32_t height;
};

template <typename vec_type>
image<vec_type> read_image(const char* path) {
  static_assert(std::is_same<vec_type, uchar4>::value || std::is_same<vec_type, float4>::value,
                "Only uchar4 and float4 images are supported");

  constexpr auto get_load_func = [&]() {
    if constexpr (std::is_same<vec_type, uchar4>::value) {
      return stbi_load;
    } else {
      return stbi_loadf;
    }
  };
  constexpr auto load_func = get_load_func();

  int width, height;
  void* image_data_ptr = load_func(path, &width, &height, nullptr, STBI_rgb_alpha);

  if (!image_data_ptr) {
    throw ImageException("Invalid image " + std::string(path));
  }

  vec_type* image_ptr = reinterpret_cast<vec_type*>(image_data_ptr);

  std::vector<vec_type> image_data(image_ptr, image_ptr + width * height);
  stbi_image_free(image_data_ptr);

  return { image_data, static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
}

image<uchar4> read_image(const uint8_t* data, int length);

template <typename vec_type>
void write_image(const char* path, const image<vec_type>& im) {
  static_assert(std::is_same<vec_type, uchar4>::value, "Only uchar4 images are supported");
  int success = stbi_write_jpg(path, im.width, im.height, STBI_rgb_alpha, im.data.data(), 100);

  if (!success) {
    throw ImageException("Failed to save image " + std::string(path));
  }
}

template <typename vec_type>
image<vec_type> resize_image(const image<vec_type>& in, uint32_t width, uint32_t height) {
  static_assert(std::is_same<vec_type, uchar4>::value || std::is_same<vec_type, float4>::value,
                "Only uchar4 and float4 images are supported");

  constexpr auto get_resize_func = [&]() {
    if constexpr (std::is_same<vec_type, uchar4>::value) {
      return stbir_resize_uint8;
    } else {
      return stbir_resize_float;
    }
  };
  constexpr auto resize_func = get_resize_func();

  using vec_comp_type =
    typename std::conditional<std::is_same<vec_type, uchar4>::value, uint8_t, float>::type;

  std::vector<vec_type> resized_image_data(width * height);
  int success = resize_func(
    reinterpret_cast<const vec_comp_type*>(in.data.data()), in.width, in.height, 0,
    reinterpret_cast<vec_comp_type*>(resized_image_data.data()), width, height, 0, STBI_rgb_alpha);

  if (!success) {
    throw ImageException("Failed to resize image");
  }

  return { resized_image_data, width, height };
}

}

#endif // IMAGEUTILS_HPP
