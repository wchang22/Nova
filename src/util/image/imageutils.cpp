#include "imageutils.h"
#include "util/exception/exception.h"

#include <stdexcept>

#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>

namespace image_utils {
  image read_image(const char* path) {
    int width, height;
    uint8_t* image_data_ptr = stbi_load(path, &width, &height, nullptr, STBI_rgb_alpha);

    if (!image_data_ptr) {
      throw std::invalid_argument("Invalid image " + std::string(path));
    }

    std::vector<uint8_t> image_data(image_data_ptr,
                                    image_data_ptr + width * height * STBI_rgb_alpha);
    stbi_image_free(image_data_ptr);

    return { image_data, static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
  }

  void write_image(const char* path, const image& im) {
    int success = stbi_write_jpg(path, im.width, im.height, STBI_rgb_alpha, im.data.data(), 100);

    if (!success) {
      throw ImageException("Failed to save image " + std::string(path));
    }
  }

  image resize_image(const image& in, uint32_t width, uint32_t height) {
    std::vector<uint8_t> resized_image_data(width * height * STBI_rgb_alpha);
    int success = stbir_resize_uint8(in.data.data(), in.width, in.height, 0,
                                     resized_image_data.data(), width, height, 0, STBI_rgb_alpha);

    if (!success) {
      throw ImageException("Failed to resize image");
    }

    return { resized_image_data, width, height };
  }
}
