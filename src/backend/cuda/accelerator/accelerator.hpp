#ifndef CUDA_ACCELERATOR_HPP
#define CUDA_ACCELERATOR_HPP

#include <cuda_runtime.h>

#include "core/scene_parser.hpp"
#include "backend/common/types/types.hpp"
#include "backend/common/utils/utils.hpp"
#include "backend/cuda/types/types.hpp"
#include "backend/cuda/entry.hpp"
#include "util/exception/exception.hpp"

#define RESOLVE_KERNEL(kernel) kernel

class Accelerator {
public:
  Accelerator(const SceneParser& scene_parser);

  void add_kernel(const std::string& kernel_name) {
    (void) kernel_name;
  }

  template<typename Kernel, typename... Args>
  void call_kernel(const Kernel& kernel, uint2 global_dims, uint2 local_dims, Args&&... args) {
    align_dims(global_dims, local_dims);
    kernel(global_dims, local_dims, kernel_constants, std::forward<Args>(args).data()...);
    CUDA_CHECK(cudaPeekAtLastError())
    CUDA_CHECK(cudaDeviceSynchronize())
  }
  
  template<typename T>
  Image2DRead<T> create_image2D_read(ImageChannelOrder channel_order, ImageChannelType channel_type,
                                     AddressMode address_mode, FilterMode filter_mode,
                                     bool normalized_coords, size_t width, size_t height,
                                     std::vector<T>& data) const {
    (void) channel_order;
    (void) channel_type;
    if (data.empty() || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DRead");
    }
    return Image2DRead(address_mode, filter_mode, normalized_coords, width, height, data);
  }

  template<typename T>
  Image2DRead<T> create_image2D_read(ImageChannelOrder channel_order, ImageChannelType channel_type,
                                     AddressMode address_mode, FilterMode filter_mode,
                                     bool normalized_coords, size_t width, size_t height) const {
    (void) channel_order;
    (void) channel_type;
    if (width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DRead");
    }
    return Image2DRead<T>(address_mode, filter_mode, normalized_coords, width, height);
  }

  template<typename T>
  Image2DWrite<T> create_image2D_write(ImageChannelOrder channel_order,
                                       ImageChannelType channel_type,
                                       size_t width, size_t height) const {
    (void) channel_order;
    (void) channel_type;
    if (width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DWrite");
    }
    return Image2DWrite<T>(width, height);
  }

  template<typename T>
  Image2DArray<T> create_image2D_array(ImageChannelOrder channel_order,
                                       ImageChannelType channel_type, AddressMode address_mode,
                                       FilterMode filter_mode, bool normalized_coords,
                                       size_t array_size, size_t width, size_t height, 
                                       std::vector<T>& data) const {
    (void) channel_order;
    (void) channel_type;
    if (data.empty() || array_size == 0 || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DArray");
    }
    return Image2DArray(address_mode, filter_mode, normalized_coords,
                        array_size, width, height, data);
  }

  template<typename T>
  std::vector<T> read_image2D(const Image2DWrite<T>& image, size_t width, size_t height) const {
    return image.read(width, height);
  }

  template<typename T>
  void copy_image2D(Image2DRead<T>& dst, const Image2DWrite<T>& src,
                    size_t width, size_t height) const {
    (void) width;
    (void) height;
    dst.copy_from(src);
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, std::vector<T>& data) const {
    (void) mem_flags;
    if (data.empty()) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return Buffer(data.size(), data.data());
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, size_t length) const {
    (void) mem_flags;
    if (length == 0) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return Buffer<T>(length);
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, T data) const {
    (void) mem_flags;
    return Buffer(1, &data);
  }

  template<typename T>
  void fill_buffer(Buffer<T>& buf, size_t length, const T& t) const {
    buf.fill(length, t);
  }

  template<typename T>
  void write_buffer(Buffer<T>& buf, const std::vector<T>& v) const {
    buf.write(v);
  }

  template<typename T>
  void write_buffer(Buffer<T>& buf, const T& t) const {
    buf.write(t);
  }

  template<typename T>
  std::vector<T> read_buffer(const Buffer<T>& buf, size_t length) const {
    return buf.read(length);
  }

  template<typename T>
  T read_buffer(const Buffer<T>& buf) const {
    return buf.read();
  }

  template<typename T, typename... Args>
  Wrapper<T> create_wrapper(Args&&... args) const {
    return Wrapper<T>(std::forward<Args>(args)...);
  }

private:
  KernelConstants kernel_constants;
};

#endif // CUDA_ACCELERATOR_HPP
