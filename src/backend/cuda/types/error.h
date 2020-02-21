#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

struct Error : public std::runtime_error {
  Error(cudaError_t error);
  cudaError_t err() const;

  cudaError_t error;
};

const char* get_error_string(cudaError_t code);

#define CUDA_CHECK(result) \
  if (result != cudaSuccess) { \
    std::cerr << __FILE__ << ": line " << __LINE__ << " "; \
    throw Error(result); \
  }

#endif // CUDA_ERROR_H
