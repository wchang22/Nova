#ifndef OPENCL_ERROR_HPP
#define OPENCL_ERROR_HPP

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

using Error = cl::Error;

const char* get_error_string(cl_int error);

}

#endif // OPENCL_ERROR_HPP
