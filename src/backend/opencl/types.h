#ifndef OPENCL_TYPES_H
#define OPENCL_TYPES_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

using float2 = cl_float2;
using float3 = cl_float3;
using float4 = cl_float4;

using Image2D = cl::Image2D;
using Image2DArray = cl::Image2DArray;
using Buffer = cl::Buffer;

enum class MemFlags {
  READ_ONLY = CL_MEM_READ_ONLY,
  WRITE_ONLY = CL_MEM_WRITE_ONLY,
  READ_WRITE = CL_MEM_READ_WRITE,
};

enum class ImageChannelOrder {
  RGB = CL_RGB,
  RGBA = CL_RGBA,
  R = CL_R,
};

enum class ImageChannelType {
  UINT8 = CL_UNSIGNED_INT8,
  UINT32 = CL_UNSIGNED_INT32,
  INT8 = CL_SIGNED_INT8,
  INT32 = CL_SIGNED_INT32,
  FLOAT = CL_FLOAT,
};

using Error = cl::Error;

#endif // OPENCL_TYPES_H