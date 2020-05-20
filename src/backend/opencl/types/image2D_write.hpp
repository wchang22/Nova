#ifndef OPENCL_IMAGE2D_WRITE_HPP
#define OPENCL_IMAGE2D_WRITE_HPP

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

#include "backend/opencl/types/image2D.hpp"

namespace nova {

template <typename T>
class Image2DWrite : public Image2D<T> {
public:
  Image2DWrite() = default;

  template <typename... Args>
  Image2DWrite(Args&&... args) : Image2D<T>(std::forward<Args>(args)...) {}

  Image2DWrite(const Image2DWrite& other) : Image2D<T>(other.image) {}
  Image2DWrite(Image2DWrite& other) : Image2D<T>(other.image) {}

  Image2DWrite& operator=(const Image2DWrite& other) {
    this->image = other.image;
    return *this;
  }
  Image2DWrite& operator=(Image2DWrite& other) {
    this->image = other.image;
    return *this;
  }

  Image2DWrite(Image2DWrite&& other) : Image2D<T>(std::move(other.image)) {}
  Image2DWrite& operator=(Image2DWrite&& other) {
    std::swap(this->image, other.image);
    return *this;
  }

  ~Image2DWrite() = default;

  const cl::Image2D& data() const { return this->image; }
  cl::Image2D& data() { return this->image; }
};

}

#endif // OPENCL_IMAGE2D_WRITE_HPP