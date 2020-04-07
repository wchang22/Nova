#include <algorithm>
#include <cmath>
#include <numeric>
#include <stb_image.h>

#include "material_loader.hpp"
#include "util/profiling/profiling.hpp"

namespace nova {

MaterialLoader::MaterialLoader() { stbi_set_flip_vertically_on_load(true); }

int MaterialLoader::load_material(const char* path, bool srgb) {
  image_utils::image<uchar4> im = image_utils::read_image<uchar4>(path);
  if (srgb) {
    constexpr auto gamma_correct = [](uint8_t x) -> uint8_t {
      return std::pow(x / 255.0f, 2.2f) * 255.0f;
    };
    std::for_each(im.data.begin(), im.data.end(), [&](uchar4& pixel) {
      pixel = { gamma_correct(x(pixel)), gamma_correct(y(pixel)), gamma_correct(z(pixel)),
                w(pixel) };
    });
  }
  materials.emplace_back(im);
  return static_cast<int>(materials.size() - 1);
}

void MaterialLoader::clear() { materials.clear(); }

MaterialData MaterialLoader::build() const {
  PROFILE_SCOPE("Resize Materials");

  if (materials.empty()) {
    return {};
  }

  // Find the average image width and height
  uint32_t width = std::accumulate(materials.cbegin(), materials.cend(), 0U,
                                   [](uint32_t sum, const auto& im) {
                                     return sum + im.width;
                                   }) /
                   materials.size();
  uint32_t height = std::accumulate(materials.cbegin(), materials.cend(), 0U,
                                    [](uint32_t sum, const auto& im) {
                                      return sum + im.height;
                                    }) /
                    materials.size();

  std::vector<float4> images_data;
  images_data.reserve(width * height * materials.size());

  // Resize all images so we can put them in a uniform array
  for (const auto& material : materials) {
    image_utils::image<uchar4> resized_image;
    if (material.width == width && material.height == height) {
      resized_image = material;
    } else {
      resized_image = image_utils::resize_image(material, width, height);
    }
    std::transform(resized_image.data.begin(), resized_image.data.end(),
                   std::back_inserter(images_data), [](const uchar4& v) -> float4 {
                     constexpr auto uchar_to_float = [](uint8_t x) {
                       return static_cast<float>(x) / 255.0f;
                     };
                     return { uchar_to_float(x(v)), uchar_to_float(y(v)), uchar_to_float(z(v)),
                              uchar_to_float(w(v)) };
                   });
  }

  return { images_data, width, height, materials.size() };
}

}