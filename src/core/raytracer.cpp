#include <stb_image.h>
#include <sstream>
#include <filesystem>

#include "raytracer.h"
#include "util/utils.h"
#include "util/exception/exception.h"
#include "util/image/imageutils.h"
#include "util/profiling/profiling.h"
#include "util/kernel/kernelutils.h"
#include "configuration.h"

Raytracer::Raytracer(uint32_t width, uint32_t height)
  : width(width), height(height),
    image_buf(width * height * sizeof(ivec4)),
    camera(CAMERA_POSITION, CAMERA_FORWARD, CAMERA_UP, width, height, CAMERA_FOVY),
    model(MODEL_PATH, intersectables),
    context(DEVICE_TYPE),
    device(context.getInfo<CL_CONTEXT_DEVICES>().front()),
    queue(context),
    program(context, utils::read_file(KERNEL_PATH)),
    image(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height)
{
  try {
    std::stringstream build_args;
    build_args << "-cl-std=CL2.0";
    build_args << " -I" << KERNELS_PATH;
    program.build(build_args.str().c_str());
  } catch (...) {
    throw KernelException(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
  }

  kernel = cl::Kernel(program, "raytrace");
}

void Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  PROFILE_SECTION_START("Build data");
  cl::Buffer triangle_buf, material_buf, bvh_buf;

  auto ec = camera.get_eye_coords();
  intersectables.build_buffers(context, triangle_buf, material_buf, bvh_buf);
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Raytrace profile");
  for (int i = 0; i < NUM_PROFILE_ITERATIONS; i++) {
    PROFILE_SCOPE("Raytrace profile loop");

    PROFILE_SECTION_START("Enqueue kernel");
    kernel_utils::set_args(kernel, image, ec, triangle_buf, material_buf, bvh_buf);
    queue.enqueueNDRangeKernel(kernel,
                               cl::NDRange(0, 0), cl::NDRange(width, height), cl::NDRange(16, 16));
    queue.finish();
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Read image");
    queue.enqueueReadImage(image, true, { 0, 0, 0 }, { width, height, 1 },
                          0, 0, image_buf.data());
    PROFILE_SECTION_END();
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write image");
  std::string image_out_name = std::filesystem::path(MODEL_PATH).stem().string() + ".jpg";
  image_utils::write_image(image_out_name.c_str(), width, height, image_buf);
  PROFILE_SECTION_END();
}
