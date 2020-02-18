#ifndef OPENCL_IMAGE2D_H
#define OPENCL_IMAGE2D_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

using Image2D = cl::Image2D;

#endif // OPENCL_IMAGE2D_H