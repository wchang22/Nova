#ifndef OPENCL_IMAGE2D_READWRITE_HPP
#define OPENCL_IMAGE2D_READWRITE_HPP

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
class Image2DReadWrite : public Image2D<T> {
public:
  Image2DReadWrite() = default;

  template <typename... Args>
  Image2DReadWrite(Args&&... args) : Image2D<T>(cl::Image2D(std::forward<Args>(args)...)) {}

  Image2DReadWrite(const Image2DReadWrite& other) : Image2D<T>(other.image) {}
  Image2DReadWrite(Image2DReadWrite& other) : Image2D<T>(other.image) {}

  Image2DReadWrite& operator=(const Image2DReadWrite& other) {
    this->image = other.image;
    return *this;
  }
  Image2DReadWrite& operator=(Image2DReadWrite& other) {
    this->image = other.image;
    return *this;
  }

  Image2DReadWrite(Image2DReadWrite&& other) : Image2D<T>(std::move(other.image)) {}
  Image2DReadWrite& operator=(Image2DReadWrite&& other) {
    std::swap(this->image, other.image);
    return *this;
  }

  ~Image2DReadWrite() = default;

  struct ReadAccessor {
    cl::Image2D& image;

    const cl::Image2D& data() const { return image; }
    cl::Image2D& data() { return image; }
  };

  struct WriteAccessor {
    cl::Image2D& image;

    const cl::Image2D& data() const { return image; }
    cl::Image2D& data() { return image; }
  };

  ReadAccessor read() { return { this->image }; }
  WriteAccessor write() { return { this->image }; }
};

}

#endif // OPENCL_IMAGE2D_READWRITE_HPP