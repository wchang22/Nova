#include "accelerator.h"
#include "util/file/fileutils.h"
#include "util/exception/exception.h"
#include "backend/opencl/entry.h"
#include "constants.h"

Accelerator::Accelerator(const SceneParser& scene_parser)
  : context(CL_DEVICE_TYPE_GPU)
{
  cl::Device device(context.getInfo<CL_CONTEXT_DEVICES>().front());
  queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  program = cl::Program(context, file_utils::read_file(KERNEL_PATH));

  const auto [ default_diffuse, default_metallic, default_roughness, default_ambient_occlusion ]
    = scene_parser.get_shading_default_settings();
  const auto [ light_position, light_intensity ] = scene_parser.get_light_settings();
  const unsigned int ray_recursion_depth = scene_parser.get_ray_recursion_depth();

  try {
    std::stringstream build_args;
    build_args
      << " -cl-fast-relaxed-math -cl-mad-enable"
      << " -I" << KERNELS_PATH_STR"opencl"
      << " -D" << STRINGIFY(TRIANGLES_PER_LEAF_BITS) << "=" << TRIANGLES_PER_LEAF_BITS
      << " -DDEFAULT_DIFFUSE=" << "(float3)("
        << default_diffuse.x << "," << default_diffuse.y << "," << default_diffuse.z << ")"
      << " -DDEFAULT_METALLIC=" << default_metallic
      << " -DDEFAULT_ROUGHNESS=" << default_roughness
      << " -DDEFAULT_AMBIENT_OCCLUSION=" << default_ambient_occlusion
      << " -DLIGHT_POSITION=" << "(float3)("
        << light_position.x << "," << light_position.y << "," << light_position.z << ")"
      << " -DLIGHT_INTENSITY=" << "(float3)("
        << light_intensity.x << "," << light_intensity.y << "," << light_intensity.z << ")"
      << " -DRAY_RECURSION_DEPTH=" << ray_recursion_depth;
    program.build(build_args.str().c_str());
  } catch (...) {
    throw KernelException(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
  }
}

void Accelerator::add_kernel(const std::string& kernel_name) {
  kernel_map[kernel_name] = cl::Kernel(program, kernel_name.c_str());
}
