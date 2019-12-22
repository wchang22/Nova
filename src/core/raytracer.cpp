#include <stb_image.h>
#include <sstream>

#include "raytracer.h"
#include "util/utils.h"
#include "util/exception/exception.h"
#include "util/image/imageutils.h"
#include "util/profiling/profiling.h"
#include "util/kernel/kernelutils.h"

#ifndef KERNELS_PATH
  #define KERNELS_PATH
#endif
#ifndef ASSETS_PATH
  #define ASSETS_PATH
#endif

#ifdef NDEBUG
  #define NUM_PROFILE_ITERATIONS 100
#else
  #define NUM_PROFILE_ITERATIONS 1
#endif

constexpr cl_device_type DEVICE_TYPE = CL_DEVICE_TYPE_GPU;
static const char* IMAGE_OUT_NAME = "raytrace.jpg";
static const char* MODEL_PATH = ASSETS_PATH"aircraft/aircraft.obj";
static const char* KERNEL_PATH = KERNELS_PATH"raytrace.cl";

constexpr vec3 CAMERA_POSITION(-4, 2.8, 5);
constexpr vec3 CAMERA_FORWARD(1, -0.5, -1);
constexpr vec3 CAMERA_UP(0, 1, 0);
constexpr int CAMERA_FOVY = 45;

using namespace glm;

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
  std::pair<cl::Buffer, size_t> triangle_buf;
  cl::Buffer mat_buf, bvh_buf;

  auto& [triangle_data, num_triangles] = triangle_buf;
  auto ec = camera.get_eye_coords();
  intersectables.build_buffers(context, triangle_buf, mat_buf, bvh_buf);
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Raytrace profile");
  for (int i = 0; i < NUM_PROFILE_ITERATIONS; i++) {
    PROFILE_SCOPE("Raytrace profile loop");

    PROFILE_SECTION_START("Enqueue kernel");
    kernel_utils::set_args(kernel, image, ec, triangle_data, num_triangles, mat_buf, bvh_buf);
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
  image_utils::write_image(IMAGE_OUT_NAME, width, height, image_buf);
  PROFILE_SECTION_END();
}
