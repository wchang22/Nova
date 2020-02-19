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

template<typename T>
class Image2D {
public:
  Image2D() = default;

  template <typename... Args>
  Image2D(Args&&... args) : image(std::forward<Args>(args)...) {}

  Image2D(const Image2D& other) : image(other.image) {}
  Image2D(Image2D& other) : image(other.image) {}
  
  Image2D& operator=(const Image2D& other) {
    image = other.image;
    return *this;
  }
  Image2D& operator=(Image2D& other) { 
    image = other.image;
    return *this;
  }

  Image2D(Image2D&& other) : image(std::move(other.image)) {}
  Image2D& operator=(Image2D&& other) {
    std::swap(image, other.image);
    return *this;
  }

  ~Image2D() = default;

  const cl::Image2D& data() const { return image; }
  cl::Image2D& data() { return image; }

private:
  cl::Image2D image;
};

#endif // OPENCL_IMAGE2D_H