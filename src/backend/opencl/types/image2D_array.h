#ifndef OPENCL_IMAGE2D_ARRAY_H
#define OPENCL_IMAGE2D_ARRAY_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

using Image2DArray = cl::Image2DArray;

#endif // OPENCL_IMAGE2D_ARRAY_H