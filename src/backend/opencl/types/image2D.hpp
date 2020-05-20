#ifndef OPENCL_IMAGE2D_HPP
#define OPENCL_IMAGE2D_HPP

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

template <typename T>
class Image2D {
public:
  Image2D() = default;

  template <typename... Args>
  Image2D(Args&&... args) : image(std::forward<Args>(args)...) {}

  Image2D(const cl::Image2D& image) : image(image) {}
  Image2D(cl::Image2D&& image) : image(std::move(image)) {}

  virtual ~Image2D() {}

protected:
  cl::Image2D image;
};

}

#endif // OPENCL_IMAGE2D_HPP