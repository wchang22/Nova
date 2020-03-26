#ifndef OPENCL_ACCELERATOR_HPP
#define OPENCL_ACCELERATOR_HPP

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

#include <unordered_map>

#include "backend/common/types/types.hpp"
#include "backend/common/utils/utils.hpp"
#include "backend/opencl/types/types.hpp"
#include "backend/opencl/utils/compatibility.hpp"
#include "backend/opencl/utils/kernel.hpp"
#include "util/exception/exception.hpp"

#define RESOLVE_KERNEL(kernel) #kernel

class Accelerator {
public:
  Accelerator();
  void add_kernel(const std::string& kernel_name);

  template <typename... Args>
  void
  call_kernel(const std::string& kernel_name, uint2 global_dims, uint2 local_dims, Args&&... args) {
    const auto& kernel_it = kernel_map.find(kernel_name);
    if (kernel_it == kernel_map.end()) {
      throw KernelException("No kernel called " + kernel_name);
    }

    align_dims(global_dims, local_dims);

    kernel_utils::set_args(kernel_it->second, std::forward<Args>(args).data()...);
    queue.enqueueNDRangeKernel(kernel_it->second, cl::NullRange,
                               cl::NDRange(global_dims.s[0], global_dims.s[1], 1),
                               cl::NDRange(local_dims.s[0], local_dims.s[1], 1));
    queue.finish();
  }

  template <typename T>
  Image2DRead<T> create_image2D_read(ImageChannelOrder channel_order,
                                     ImageChannelType channel_type,
                                     AddressMode address_mode,
                                     FilterMode filter_mode,
                                     bool normalized_coords,
                                     size_t width,
                                     size_t height,
                                     std::vector<T>& data) const {
    (void) address_mode;
    (void) filter_mode;
    (void) normalized_coords;
    if (data.empty() || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DRead");
    }
    return Image2DRead<T>(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          cl::ImageFormat(static_cast<cl_channel_order>(channel_order),
                                          static_cast<cl_channel_type>(channel_type)),
                          width, height, 0, data.data());
  }

  template <typename T>
  Image2DRead<T> create_image2D_read(ImageChannelOrder channel_order,
                                     ImageChannelType channel_type,
                                     AddressMode address_mode,
                                     FilterMode filter_mode,
                                     bool normalized_coords,
                                     size_t width,
                                     size_t height) const {
    (void) address_mode;
    (void) filter_mode;
    (void) normalized_coords;
    if (width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DRead");
    }
    return Image2DRead<T>(context, CL_MEM_READ_ONLY,
                          cl::ImageFormat(static_cast<cl_channel_order>(channel_order),
                                          static_cast<cl_channel_type>(channel_type)),
                          width, height, 0);
  }

  template <typename T>
  Image2DWrite<T> create_image2D_write(ImageChannelOrder channel_order,
                                       ImageChannelType channel_type,
                                       size_t width,
                                       size_t height) const {
    if (width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DWrite");
    }
    return Image2DWrite<T>(context, CL_MEM_WRITE_ONLY,
                           cl::ImageFormat(static_cast<cl_channel_order>(channel_order),
                                           static_cast<cl_channel_type>(channel_type)),
                           width, height);
  }

  template <typename T>
  Image2DArray<T> create_image2D_array(ImageChannelOrder channel_order,
                                       ImageChannelType channel_type,
                                       AddressMode address_mode,
                                       FilterMode filter_mode,
                                       bool normalized_coords,
                                       size_t array_size,
                                       size_t width,
                                       size_t height,
                                       std::vector<T>& data) const {
    (void) address_mode;
    (void) filter_mode;
    (void) normalized_coords;
    if (data.empty() || array_size == 0 || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DArray");
    }

    return Image2DArray<T>(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           cl::ImageFormat(static_cast<cl_channel_order>(channel_order),
                                           static_cast<cl_channel_type>(channel_type)),
                           array_size, width, height, 0, 0, data.data());
  }

  template <typename T>
  std::vector<T> read_image2D(const Image2DWrite<T>& image, size_t width, size_t height) const {
    std::vector<T> image_buf(width * height);
    queue.enqueueReadImage(image.data(), true, compat_utils::create_size_t<3>({ 0, 0, 0 }),
                           compat_utils::create_size_t<3>({ width, height, 1 }), 0, 0,
                           image_buf.data());
    return image_buf;
  }

  template <typename T>
  void
  copy_image2D(Image2DRead<T>& dst, const Image2DWrite<T>& src, size_t width, size_t height) const {
    queue.enqueueCopyImage(src.data(), dst.data(), compat_utils::create_size_t<3>({ 0, 0, 0 }),
                           compat_utils::create_size_t<3>({ 0, 0, 0 }),
                           compat_utils::create_size_t<3>({ width, height, 1 }));
    queue.finish();
  }

  template <typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, std::vector<T>& data) const {
    if (data.empty()) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return Buffer<T>(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                     data.size() * sizeof(T), data.data());
  }

  template <typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, size_t length) const {
    if (length == 0) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return Buffer<T>(context, static_cast<cl_mem_flags>(mem_flags), length * sizeof(T));
  }

  template <typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, T data) const {
    return Buffer<T>(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                     sizeof(T), &data);
  }

  template <typename T>
  void fill_buffer(Buffer<T>& buf, size_t length, const T& t) const {
    queue.enqueueFillBuffer(buf.data(), t, 0, sizeof(T) * length);
    queue.finish();
  }

  template <typename T>
  void write_buffer(Buffer<T>& buf, const std::vector<T>& v) const {
    queue.enqueueWriteBuffer(buf.data(), true, 0, sizeof(T) * v.size(), v.data());
  }

  template <typename T>
  void write_buffer(Buffer<T>& buf, const T& t) const {
    queue.enqueueWriteBuffer(buf.data(), true, 0, sizeof(T), &t);
  }

  template <typename T>
  std::vector<T> read_buffer(const Buffer<T>& buf, size_t length) const {
    std::vector<T> buf_vec(length);
    queue.enqueueReadBuffer(buf.data(), true, 0, sizeof(T) * length, buf_vec.data());
    return buf_vec;
  }

  template <typename T>
  T read_buffer(const Buffer<T>& buf) const {
    T t;
    queue.enqueueReadBuffer(buf.data(), true, 0, sizeof(T), &t);
    return t;
  }

  template <typename T, typename... Args>
  Wrapper<T> create_wrapper(Args&&... args) const {
    return Wrapper<T>(std::forward<Args>(args)...);
  }

private:
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  std::unordered_map<std::string, cl::Kernel> kernel_map;
};

#endif // OPENCL_ACCELERATOR_HPP