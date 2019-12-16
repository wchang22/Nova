#include "imageutils.h"

#include <stdexcept>

#include <stb_image.h>
#include <stb_image_write.h>

namespace image_utils {
  std::pair<std::vector<uint8_t>, std::pair<uint32_t, uint32_t>>
  read_image(const char* path) {
    int width, height;
    uint8_t* image_data = stbi_load(path, &width, &height, nullptr, STBI_rgb_alpha);

    if (!image_data) {
      throw std::invalid_argument("Invalid image " + std::string(path));
    }

    std::vector<uint8_t> image(image_data, image_data + width * height * STBI_rgb_alpha);
    stbi_image_free(image_data);

    return { image, { width, height } };
  }

  void write_image(const char* path, uint32_t width, uint32_t height,
                              const std::vector<uint8_t>& image) {
    int iwidth = static_cast<int>(width);
    int iheight = static_cast<int>(height);
    int success = stbi_write_jpg(path, iwidth, iheight, STBI_rgb_alpha, image.data(), 100);

    if (!success) {
      throw std::runtime_error("Failed to save image " + std::string(path));
    }
  }
}
