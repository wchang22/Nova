#include <numeric>
#include <stb_image.h>

#include "material_loader.h"

MaterialLoader::MaterialLoader() {
  stbi_set_flip_vertically_on_load(true);
}

int MaterialLoader::load_material(const char* path) {
  materials.emplace_back(image_utils::read_image(path));
  return static_cast<int>(materials.size() - 1);
}

MaterialData MaterialLoader::build() const {
  if (materials.empty()) {
    return {};
  }

  // Find the average image width and height
  uint32_t width = std::accumulate(materials.cbegin(), materials.cend(), 0U,
    [](uint32_t sum, const auto& im) {
      return sum + im.width;
    }) / materials.size();
  uint32_t height = std::accumulate(materials.cbegin(), materials.cend(), 0U,
    [](uint32_t sum, const auto& im) {
      return sum + im.height;
    }) / materials.size();

  std::vector<uint8_t> images_data;
  images_data.reserve(width * height * materials.size() * STBI_rgb_alpha);

  // Resize all images so we can put them in a uniform array
  for (const auto& material : materials) {
    image_utils::image resized_image;
    if (material.width == width && material.height == height) {
      resized_image = material;
    } else {
      resized_image = image_utils::resize_image(material, width, height);
    }
    images_data.insert(images_data.end(),
                        std::make_move_iterator(resized_image.data.begin()),
                        std::make_move_iterator(resized_image.data.end()));
  }

  return {
    images_data,
    width,
    height,
    materials.size()
  };
}
