#ifndef OPENCL_IMAGE2D_READ_HPP
#define OPENCL_IMAGE2D_READ_HPP

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
class Image2DRead {
public:
  Image2DRead() = default;

  template <typename... Args>
  Image2DRead(Args&&... args) : image(std::forward<Args>(args)...) {}

  Image2DRead(const Image2DRead& other) : image(other.image) {}
  Image2DRead(Image2DRead& other) : image(other.image) {}

  Image2DRead& operator=(const Image2DRead& other) {
    image = other.image;
    return *this;
  }
  Image2DRead& operator=(Image2DRead& other) {
    image = other.image;
    return *this;
  }

  Image2DRead(Image2DRead&& other) : image(std::move(other.image)) {}
  Image2DRead& operator=(Image2DRead&& other) {
    std::swap(image, other.image);
    return *this;
  }

  ~Image2DRead() = default;

  const cl::Image2D& data() const { return image; }
  cl::Image2D& data() { return image; }

private:
  cl::Image2D image;
};

}

#endif // OPENCL_IMAGE2D_READ_HPP