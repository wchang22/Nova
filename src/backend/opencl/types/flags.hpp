#ifndef OPENCL_FLAGS_HPP
#define OPENCL_FLAGS_HPP

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

namespace nova {

enum class MemFlags {
  READ_ONLY = CL_MEM_READ_ONLY,
  WRITE_ONLY = CL_MEM_WRITE_ONLY,
  READ_WRITE = CL_MEM_READ_WRITE
};

enum class ImageChannelOrder { RGB = CL_RGB, RGBA = CL_RGBA, R = CL_R };

enum class ImageChannelType {
  UINT8 = CL_UNSIGNED_INT8,
  UINT32 = CL_UNSIGNED_INT32,
  INT8 = CL_SIGNED_INT8,
  INT32 = CL_SIGNED_INT32,
  FLOAT = CL_FLOAT
};

enum class AddressMode {
  WRAP,
  CLAMP,
  MIRROR,
  BORDER,
};

enum class FilterMode {
  NEAREST,
  LINEAR,
};

}

#endif // OPENCL_FLAGS_HPP