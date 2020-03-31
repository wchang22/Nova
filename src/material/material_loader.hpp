#ifndef MATERIALS_HPP
#define MATERIALS_HPP

#include "util/image/imageutils.hpp"

namespace nova {

struct MaterialData {
  std::vector<uchar4> data;
  uint32_t width;
  uint32_t height;
  size_t num_materials;
};

class MaterialLoader {
public:
  MaterialLoader();

  int load_material(const char* path, bool srgb = false);
  void clear();
  MaterialData build() const;

private:
  std::vector<image_utils::image> materials;
};

}

#endif // MATERIALS_HPP
