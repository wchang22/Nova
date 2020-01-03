#ifndef KERNEL_H
#define KERNEL_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

namespace kernel_utils {
  void set_args_helper(cl::Kernel& kernel, uint32_t i) {
    (void) kernel;
    (void) i;
  }

  template <typename Arg, typename ...Args>
  void set_args_helper(cl::Kernel& kernel, uint32_t i, Arg first, Args... rest) {
    kernel.setArg(i, first);
    set_args_helper(kernel, i + 1, rest...);
  }

  template <typename ...Args>
  void set_args(cl::Kernel& kernel, Args... args) {
    set_args_helper(kernel, 0, args...);
  }
}

#endif // KERNEL_H
