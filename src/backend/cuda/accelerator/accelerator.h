#ifndef CUDA_ACCELERATOR_H
#define CUDA_ACCELERATOR_H

#include <cuda_runtime.h>

#include "core/scene_parser.h"
#include "backend/common/types/types.h"
#include "backend/cuda/types/types.h"
#include "backend/cuda/entry.h"
#include "util/exception/exception.h"

#define ADD_KERNEL(accel, kernel_name)
#define CALL_KERNEL(accel, kernel, global_size, local_size, ...) \
  kernel(global_size, local_size, __VA_ARGS__);

struct KernelConstants {
  uint32_t triangle_per_leaf_bits;
  float3 default_diffuse;
  float default_metallic;
  float default_roughness;
  float default_ambient_occlusion;
  float3 light_position;
  float3 light_intensity;
  uint32_t ray_recursion_depth;
};

class Accelerator {
public:
  Accelerator(const SceneParser& scene_parser);
  
  template<typename T>
  Image2D<T> create_image2D(MemFlags mem_flags, ImageChannelOrder channel_order,
                            ImageChannelType channel_type, AddressMode address_mode,
                            FilterMode filter_mode, bool normalized_coords,
                            size_t width, size_t height, std::vector<T>& data) const {
    (void) mem_flags;
    (void) channel_order;
    (void) channel_type;
    if (data.empty() || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2D");
    }
    return Image2D(address_mode, filter_mode, normalized_coords, width, height, data.data());
  }

  template<typename T>
  Image2D<T> create_image2D(MemFlags mem_flags, ImageChannelOrder channel_order,
                            ImageChannelType channel_type, AddressMode address_mode,
                            FilterMode filter_mode, bool normalized_coords,
                            size_t width, size_t height) const {
    (void) mem_flags;
    (void) channel_order;
    (void) channel_type;
    if (width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2D");
    }
    return Image2D<T>(address_mode, filter_mode, normalized_coords, width, height);
  }

  template<typename T>
  Image2DArray create_image2D_array(MemFlags mem_flags, ImageChannelOrder channel_order,
                                    ImageChannelType channel_type, size_t array_size, size_t width, size_t height, std::vector<T>& data) const {
    if (data.empty() || array_size == 0 || width == 0 || height == 0) {
      throw AcceleratorException("Cannot build an empty Image2DArray");
    }
  }

  Image2DArray create_image2D_array(MemFlags mem_flags, ImageChannelOrder channel_order,
                                    ImageChannelType channel_type, size_t array_size, size_t width, size_t height) const;

  template<typename T>
  std::vector<T> read_image(const Image2D<T>& image, size_t width, size_t height) const {
    std::vector<T> image_buf(width * height);
    int pitch_in_bytes = width * sizeof(T);
    CUDA_CHECK(cudaMemcpy2D(image_buf.data(), pitch_in_bytes, image.ptr(),
                            pitch_in_bytes, pitch_in_bytes, height, cudaMemcpyDeviceToHost))
    return image_buf;
  }

  template<typename T>
  Buffer<T> create_buffer(MemFlags mem_flags, T& data) const {
    (void) mem_flags;
    return Buffer(1, &data);
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
    if (length == 0) {
      throw AcceleratorException("Cannot build an empty Buffer");
    }
    return Buffer<T>(length);
  }

  template<typename T, typename... Args>
  Wrapper<T> create_wrapper(Args&&... args) const {
    return Wrapper<T>(std::forward<Args>(args)...);
  }

private:
  KernelConstants kernel_constants;
};

#endif // CUDA_ACCELERATOR_H