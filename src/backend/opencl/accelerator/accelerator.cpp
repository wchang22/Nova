#include "accelerator.hpp"
#include "backend/opencl/entry.hpp"
#include "backend/opencl/utils/build_args.hpp"
#include "constants.hpp"
#include "util/exception/exception.hpp"
#include "util/file/fileutils.hpp"

Accelerator::Accelerator(const SceneParser& scene_parser) : context(CL_DEVICE_TYPE_GPU) {
  cl::Device device(context.getInfo<CL_CONTEXT_DEVICES>().front());
  queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  program = cl::Program(context, file_utils::read_file(KERNEL_PATH));

  const auto [default_diffuse, default_metallic, default_roughness, default_ambient_occlusion] =
    scene_parser.get_shading_default_settings();
  const auto [light_position, light_intensity] = scene_parser.get_light_settings();
  const int ray_recursion_depth = scene_parser.get_ray_recursion_depth();

  try {
    BuildArgs build_args;
    build_args.add_flag("-cl-fast-relaxed-math");
    build_args.add_flag("-cl-mad-enable");
    build_args.add_include_dir(KERNELS_PATH_STR "opencl");
    build_args.add_define("TRIANGLES_PER_LEAF_BITS", TRIANGLES_PER_LEAF_BITS);
    build_args.add_define("TRIANGLE_NUM_SHIFT", TRIANGLE_NUM_SHIFT);
    build_args.add_define("TRIANGLE_OFFSET_MASK", TRIANGLE_OFFSET_MASK);
    build_args.add_define("DEFAULT_DIFFUSE", default_diffuse);
    build_args.add_define("DEFAULT_METALLIC", default_metallic);
    build_args.add_define("DEFAULT_ROUGHNESS", default_roughness);
    build_args.add_define("DEFAULT_AMBIENT_OCCLUSION", default_ambient_occlusion);
    build_args.add_define("LIGHT_POSITION", light_position);
    build_args.add_define("LIGHT_INTENSITY", light_intensity);
    build_args.add_define("RAY_RECURSION_DEPTH", ray_recursion_depth);
    program.build(build_args.str().c_str());
  } catch (...) {
    throw KernelException(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
  }
}

void Accelerator::add_kernel(const std::string& kernel_name) {
  kernel_map[kernel_name] = cl::Kernel(program, kernel_name.c_str());
}
