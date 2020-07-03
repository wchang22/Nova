#include <iostream>

#include "accelerator.hpp"
#include "backend/opencl/utils/build_args.hpp"
#include "constants.hpp"
#include "util/exception/exception.hpp"
#include "util/file/fileutils.hpp"

namespace nova {

Accelerator::Accelerator() {
  try {
    context = cl::Context(CL_DEVICE_TYPE_GPU);
  } catch (...) {
    try {
      context = cl::Context(CL_DEVICE_TYPE_CPU);
    } catch (...) {
      throw AcceleratorException("No OpenCL devices available.");
    }
  }

  cl::Device device(context.getInfo<CL_CONTEXT_DEVICES>().front());
  std::cout << "Using device " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  std::string binary = file_utils::read_file(OPENCL_KERNEL_BINARY);
  cl_int err = 0;
  program = cl::Program(clCreateProgramWithIL(context(), binary.data(), binary.length(), &err));
  if (err) {
    throw KernelException(cl::Error(err).what());
  }

  try {
    BuildArgs build_args;
    build_args.add_flag("-cl-fast-relaxed-math");
    build_args.add_flag("-cl-mad-enable");
    program.build(build_args.str().c_str());
  } catch (...) {
    throw KernelException(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
  }
}

void Accelerator::add_kernel(const std::string& kernel_name) {
  kernel_map[kernel_name] = cl::Kernel(program, kernel_name.c_str());
}

}
