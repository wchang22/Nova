#ifndef MATERIALS_H
#define MATERIALS_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

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
