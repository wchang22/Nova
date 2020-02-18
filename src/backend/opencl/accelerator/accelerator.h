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
#include "backend/opencl/types/types.h"
#include "backend/opencl/utils/kernel.h"
#include "backend/opencl/utils/compatibility.h"
#include "util/exception/exception.h"

#define ADD_KERNEL(accel, kernel) accel.add_kernel(#kernel);
#define CALL_KERNEL(accel, kernel, global_size, local_size, ...) \
  accel.call_kernel(#kernel, global_size, local_size, __VA_ARGS__);

class Accelerator {
public:
  Accelerator(const SceneParser& scene_parser);
  void add_kernel(const std::string& kernel_name);

  template<typename... GlobalDims, typename... LocalDims, typename... Args>
  void call_kernel(const std::string& kernel_name, std::tuple<GlobalDims...> global_size, 
                   std::tuple<LocalDims...> local_size, Args... args) {
    cl::Kernel& kernel = kernel_map.at(kernel_name);

    kernel_utils::set_args(kernel, args...);

    std::apply([&](auto&&... global_range) { 
      std::apply([&](auto&&... local_range) {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global_range...), cl::NDRange(local_range...));
      }, local_size);
    }, global_size);
    
    queue.finish();
  }
  
  template<typename T>
  Image2D create_image2D(MemFlags mem_flags, ImageChannelOrder channel_order,
                         ImageChannelType channel_type, size_t width, size_t height,
                         std::vector<T>& data) const {
    if (data.empty() || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2D");
    }
    return cl::Image2D(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                       cl::ImageFormat(
                         static_cast<cl_channel_order>(channel_order),
                         static_cast<cl_channel_type>(channel_type)
                       ),
                       width, height, 0, data.data());
  }

  Image2D create_image2D(MemFlags mem_flags, ImageChannelOrder channel_order,
                         ImageChannelType channel_type, size_t width, size_t height) const;

  template<typename T>
  Image2DArray create_image2D_array(MemFlags mem_flags, ImageChannelOrder channel_order,
                                    ImageChannelType channel_type, size_t array_size, size_t width, size_t height, std::vector<T>& data) const {
    if (data.empty() || array_size == 0 || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DArray");
    }

    return cl::Image2DArray(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                            cl::ImageFormat(
                              static_cast<cl_channel_order>(channel_order),
                              static_cast<cl_channel_type>(channel_type)
                            ),
                            array_size, width, height, 0, 0, data.data());
  }

  Image2DArray create_image2D_array(MemFlags mem_flags, ImageChannelOrder channel_order,
                                    ImageChannelType channel_type, size_t array_size, size_t width, size_t height) const;

  template<typename T>
  std::vector<T> read_image(const Image2D& image, size_t width, size_t height,
                            size_t num_channels) const {
    std::vector<T> image_buf(width * height * num_channels);
    queue.enqueueReadImage(image, true,
                           compat_utils::create_size_t<3>({ 0, 0, 0 }),
                           compat_utils::create_size_t<3>({ width, height, 1 }),
                           0, 0, image_buf.data());
    return image_buf;
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, T& data) const {
    return cl::Buffer(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                      sizeof(T), &data);
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, std::vector<T>& data) const {
    if (data.empty()) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return cl::Buffer(context, static_cast<cl_mem_flags>(mem_flags) | CL_MEM_COPY_HOST_PTR,
                      data.size() * sizeof(T), data.data());
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, size_t length) const {
    if (length == 0) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return cl::Buffer(context, static_cast<cl_mem_flags>(mem_flags), length * sizeof(T));
  }

private:
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  std::unordered_map<std::string, cl::Kernel> kernel_map;
};

#endif // OPENCL_ACCELERATOR_H