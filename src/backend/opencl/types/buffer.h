#ifndef OPENCL_BUFFER_H
#define OPENCL_BUFFER_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

template <typename T>
using Buffer = cl::Buffer;

#endif // OPENCL_BUFFER_H