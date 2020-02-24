#ifndef OPENCL_IMAGE2D_WRITE_H
#define OPENCL_IMAGE2D_WRITE_H

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
class Image2DWrite {
public:
  Image2DWrite() = default;

  template <typename... Args>
  Image2DWrite(Args&&... args) : image(std::forward<Args>(args)...) {}

  Image2DWrite(const Image2DWrite& other) : image(other.image) {}
  Image2DWrite(Image2DWrite& other) : image(other.image) {}
  
  Image2DWrite& operator=(const Image2DWrite& other) {
    image = other.image;
    return *this;
  }
  Image2DWrite& operator=(Image2DWrite& other) { 
    image = other.image;
    return *this;
  }

  Image2DWrite(Image2DWrite&& other) : image(std::move(other.image)) {}
  Image2DWrite& operator=(Image2DWrite&& other) {
    std::swap(image, other.image);
    return *this;
  }

  ~Image2DWrite() = default;

  const cl::Image2D& data() const { return image; }
  cl::Image2D& data() { return image; }

private:
  cl::Image2D image;
};

#endif // OPENCL_IMAGE2D_WRITE_H