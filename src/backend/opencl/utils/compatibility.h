#ifndef OPENCL_COMPATIBILITY_H
#define OPENCL_COMPATIBILITY_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

namespace compat_utils {
  // Opencl 1.x and 2.x have different definitions for size_t arrays
  #ifdef OPENCL_2
    #define cl_size_t_array std::array<size_t, N>
  #else
    #define cl_size_t_array cl::size_t<N>
  #endif

  template <int N>
  cl_size_t_array create_size_t(const std::array<size_t, N>& arr) {
    cl_size_t_array s;
    for (int i = 0; i < N; i++) {
      s[i] = arr[i];
    }
    return s;
  }
}

#endif // OPENCL_COMPATIBILITY_H
