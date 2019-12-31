#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <vector>
#include <cstdint>

namespace image_utils {
  struct image {
    std::vector<uint8_t> data;
    uint32_t width;
    uint32_t height;
  };

  image read_image(const char* path);

  void write_image(const char* path, const image& im);

  image resize_image(const image& in, uint32_t width, uint32_t height);
}

#endif // IMAGEUTILS_H
