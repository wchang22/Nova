#ifndef OPENCL_IMAGE2D_ARRAY_HPP
#define OPENCL_IMAGE2D_ARRAY_HPP

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
class Image2DArray {
public:
  Image2DArray() = default;

  template <typename... Args>
  Image2DArray(Args&&... args) : image(std::forward<Args>(args)...) {}

  Image2DArray(const Image2DArray& other) : image(other.image) {}
  Image2DArray(Image2DArray& other) : image(other.image) {}

  Image2DArray& operator=(const Image2DArray& other) {
    image = other.image;
    return *this;
  }
  Image2DArray& operator=(Image2DArray& other) {
    image = other.image;
    return *this;
  }

  Image2DArray(Image2DArray&& other) : image(std::move(other.image)) {}
  Image2DArray& operator=(Image2DArray&& other) {
    std::swap(image, other.image);
    return *this;
  }

  ~Image2DArray() = default;

  const cl::Image2DArray& data() const { return image; }
  cl::Image2DArray& data() { return image; }

private:
  cl::Image2DArray image;
};

}

#endif // OPENCL_IMAGE2D_ARRAY_HPP