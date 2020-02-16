#ifndef MATERIALS_H
#define MATERIALS_H

#include "util/image/imageutils.h"

struct MaterialData {
  std::vector<uint8_t> data;
  uint32_t width;
  uint32_t height;
  size_t num_materials;
};

class MaterialLoader {
public:
  MaterialLoader();

  int load_material(const char* path);
  MaterialData build() const;

private:
  std::vector<image_utils::image> materials;
};

#endif // MATERIALS_H
