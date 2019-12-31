#ifndef MATERIALS_H
#define MATERIALS_H

#include <CL/cl2.hpp>

#include "util/image/imageutils.h"

class MaterialLoader {
public:
  MaterialLoader();

  int load_material(const char* path);
  cl::Image2DArray build_images(const cl::Context& context);

private:
  std::vector<image_utils::image> materials;
};

#endif // MATERIALS_H
