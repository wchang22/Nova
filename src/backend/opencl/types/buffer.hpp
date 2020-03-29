#ifndef OPENCL_BUFFER_HPP
#define OPENCL_BUFFER_HPP

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
class Buffer {
public:
  Buffer() = default;

  template <typename... Args>
  Buffer(Args&&... args) : buffer(std::forward<Args>(args)...) {}

  Buffer(const Buffer& other) : buffer(other.buffer) {}
  Buffer(Buffer& other) : buffer(other.buffer) {}

  Buffer& operator=(const Buffer& other) {
    buffer = other.buffer;
    return *this;
  }
  Buffer& operator=(Buffer& other) {
    buffer = other.buffer;
    return *this;
  }

  Buffer(Buffer&& other) : buffer(std::move(other.buffer)) {}
  Buffer& operator=(Buffer&& other) {
    std::swap(buffer, other.buffer);
    return *this;
  }

  ~Buffer() = default;

  const cl::Buffer& data() const { return buffer; }
  cl::Buffer& data() { return buffer; }

private:
  cl::Buffer buffer;
};

}

#endif // OPENCL_BUFFER_HPP