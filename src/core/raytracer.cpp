#include <stb_image.h>
#include <glm/glm.hpp>

#include "raytracer.h"
#include "util/utils.h"
#include "util/exception/exception.h"
#include "util/image/imageutils.h"
#include "util/profiling/profiling.h"

#ifndef KERNEL_PATH
  #define KERNEL_PATH
#endif

constexpr cl_device_type DEVICE_TYPE = CL_DEVICE_TYPE_GPU;
const char* IMAGE_OUT_NAME = "raytrace.jpg";

using namespace glm;

Raytracer::Raytracer(uint32_t width, uint32_t height)
  : width(width), height(height),
    image_buf(width * height * sizeof(ivec4)),
    context(DEVICE_TYPE),
    device(context.getInfo<CL_CONTEXT_DEVICES>().front()),
    queue(context),
    program(context, utils::read_file(KERNEL_PATH"raytrace.cl")),
    image(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
          width, height)
{
  try {
    program.build("-cl-std=CL2.0");
  } catch (...) {
    throw KernelException(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
  }

  kernel = std::make_unique<Kernel>(program, "raytrace");
}

void Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  PROFILE_SECTION_START("Enqueue kernel");
  (*kernel)(cl::EnqueueArgs(queue, cl::NDRange(width, height)), image);
  queue.finish();
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Read image");
  queue.enqueueReadImage(image, true, { 0, 0, 0 }, { width, height, 1 },
                         0, 0, image_buf.data());
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write image");
  image_utils::write_image(IMAGE_OUT_NAME, width, height, image_buf);
  PROFILE_SECTION_END();
}