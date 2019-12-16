#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <vector>
#include <cstdint>

namespace image_utils {
  std::pair<std::vector<uint8_t>, std::pair<uint32_t, uint32_t>> read_image(const char* path);

  void write_image(const char* path, uint32_t width, uint32_t height,
                   const std::vector<uint8_t>& image);
}

#endif // IMAGEUTILS_H
