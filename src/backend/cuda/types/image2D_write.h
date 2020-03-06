#ifndef CUDA_IMAGE2D_WRITE_H
#define CUDA_IMAGE2D_WRITE_H

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "backend/cuda/types/error.h"
#include "backend/cuda/types/flags.h"
#include "backend/cuda/types/image2D.h"

template<typename T>
class Image2DWrite : public Image2D<T> {
public:
  Image2DWrite(size_t width, size_t height)
    : Image2D<T>(width, height)
  {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CUDA_CHECK(cudaMallocArray(&this->buffer, &channel_desc, width, height,
                               cudaArraySurfaceLoadStore))

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = this->buffer;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res_desc))
  }

  ~Image2DWrite() {
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(this->buffer);
  }

  cudaSurfaceObject_t& data() { return surf; };

  std::vector<T> read() const {
    std::vector<T> image_data(this->width * this->height);
    CUDA_CHECK(cudaMemcpy2DFromArray(image_data.data(), this->width * sizeof(T), this->buffer, 0, 0,
                                     this->width * sizeof(T), this->height, cudaMemcpyDeviceToHost))
    return image_data;
  }

private:
  cudaSurfaceObject_t surf;
};

#endif // CUDA_IMAGE2D_WRITE_H