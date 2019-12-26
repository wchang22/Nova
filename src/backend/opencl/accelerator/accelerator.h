#ifndef OPENCL_ACCELERATOR_H
#define OPENCL_ACCELERATOR_H

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

#include "core/scene_parser.h"
#include "backend/common/types/types.h"
#include "backend/opencl/types/types.h"
#include "backend/opencl/utils/kernel.h"
#include "backend/opencl/utils/compatibility.h"
#include "util/exception/exception.h"

#define ADD_KERNEL(accel, kernel) accel.add_kernel(#kernel);
#define REGISTER_KERNEL(kernel)
#define CALL_KERNEL(accel, kernel, global_size, ...) \
  accel.call_kernel(#kernel, global_size, __VA_ARGS__);

class Accelerator {
public:
  Accelerator(const SceneParser& scene_parser);
  void add_kernel(const std::string& kernel_name);

  template<typename... Args>
  void call_kernel(const std::string& kernel_name, uint3 global_dims, Args&&... args) {
    const auto& kernel_it = kernel_map.find(kernel_name);
    if (kernel_it == kernel_map.end()) {
      throw KernelException("No kernel called " + kernel_name);
    }

    kernel_utils::set_args(kernel_it->second, std::forward<Args>(args).data()...);
    queue.enqueueNDRangeKernel(kernel_it->second, cl::NullRange,
                               cl::NDRange(global_dims.s[0], global_dims.s[1], global_dims.s[2]),
                               cl::NullRange);
    
    queue.finish();
  }
  
  template<typename T>
  Image2DRead<T> create_image2D_read(ImageChannelOrder channel_order, ImageChannelType channel_type,                                   AddressMode address_mode, FilterMode filter_mode,
                                     bool normalized_coords, size_t width, size_t height, 
                                     std::vector<T>& data) const {
    (void) address_mode;
    (void) filter_mode;
    (void) normalized_coords;
    if (data.empty() || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DRead");
    }
    return Image2DRead<T>(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          cl::ImageFormat(
                            static_cast<cl_channel_order>(channel_order),
                            static_cast<cl_channel_type>(channel_type)
                          ),
                          width, height, 0, data.data());
  }

  template<typename T>
  Image2DWrite<T> create_image2D_write(ImageChannelOrder channel_order,
                                       ImageChannelType channel_type,
                                       size_t width, size_t height) const {
    if (width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DWrite");
    }
    return Image2DWrite<T>(context, CL_MEM_WRITE_ONLY,
                           cl::ImageFormat(
                             static_cast<cl_channel_order>(channel_order),
                             static_cast<cl_channel_type>(channel_type)
                           ),
                           width, height);
  }

  template<typename T>
  Image2DArray<T> create_image2D_array(ImageChannelOrder channel_order,
                                       ImageChannelType channel_type, AddressMode address_mode,
                                       FilterMode filter_mode, bool normalized_coords,
                                       size_t array_size, size_t width, size_t height,
                                       std::vector<T>& data) const {
    (void) address_mode;
    (void) filter_mode;
    (void) normalized_coords;
    if (data.empty() || array_size == 0 || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DArray");
    }

    return Image2DArray<T>(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           cl::ImageFormat(
                             static_cast<cl_channel_order>(channel_order),
                             static_cast<cl_channel_type>(channel_type)
                           ),
                           array_size, width, height, 0, 0, data.data());
  }

  template<typename T>
  std::vector<T> read_image(const Image2DWrite<T>& image, size_t width, size_t height) const {
    std::vector<T> image_buf(width * height);
    queue.enqueueReadImage(image.data(), true,
                           compat_utils::create_size_t<3>({ 0, 0, 0 }),
                           compat_utils::create_size_t<3>({ width, height, 1 }),
                           0, 0, image_buf.data());
    return image_buf;
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, T& data) const {
    return Buffer<T>(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                     sizeof(T), &data);
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, std::vector<T>& data) const {
    if (data.empty()) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return Buffer<T>(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                     data.size() * sizeof(T), data.data());
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, size_t length) const {
    if (length == 0) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return Buffer<T>(context, static_cast<cl_mem_flags>(mem_flags), length * sizeof(T));
  }

  template<typename T, typename... Args>
  Wrapper<T> create_wrapper(Args&&... args) const {
    return Wrapper<T>(std::forward<Args>(args)...);
  }

private:
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  std::unordered_map<std::string, cl::Kernel> kernel_map;
};

#endif // OPENCL_ACCELERATOR_H