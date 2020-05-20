#include "imageutils.hpp"

namespace image_utils {

image<uchar4> read_image(const uint8_t* data, int length) {
  int width, height;
  uint8_t* image_data_ptr =
    stbi_load_from_memory(data, length, &width, &height, nullptr, STBI_rgb_alpha);

  uchar4* image_ptr = reinterpret_cast<uchar4*>(image_data_ptr);

  std::vector<uchar4> image_data(image_ptr, image_ptr + width * height);
  stbi_image_free(image_data_ptr);

  return { image_data, static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
}

}