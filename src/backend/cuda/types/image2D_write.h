#ifndef CUDA_IMAGE2D_H
#define CUDA_IMAGE2D_H

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "backend/cuda/types/error.h"
#include "backend/cuda/types/flags.h"

template<typename T>
class Image2DWrite {
public:
  Image2DWrite(size_t width, size_t height)
    : width(width), height(height)
  {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CUDA_CHECK(cudaMallocArray(&buffer, &channel_desc, width, height, cudaArraySurfaceLoadStore))

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = buffer;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res_desc))
  }

  ~Image2DWrite() {
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(buffer);
  }

  cudaSurfaceObject_t& data() { return surf; };

  std::vector<T> read() const {
    std::vector<T> image_data(width * height);
    CUDA_CHECK(cudaMemcpy2DFromArray(image_data.data(), width * sizeof(T), buffer, 0, 0,
                                     width * sizeof(T), height, cudaMemcpyDeviceToHost))
    return image_data;
  }

private:
  cudaSurfaceObject_t surf;
  cudaArray_t buffer;
  size_t width;
  size_t height;
};

#endif // CUDA_IMAGE2D_H