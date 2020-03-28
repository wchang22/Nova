#ifndef CUDA_ERROR_HPP
#define CUDA_ERROR_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

struct Error : public std::runtime_error {
  Error(cudaError_t error);
  cudaError_t err() const;

  cudaError_t error;
};

const char* get_error_string(cudaError_t code);

#define CUDA_CHECK(result)                                                            \
  if (result != cudaSuccess) {                                                        \
    std::cerr << __FILE__ << ": line " << __LINE__ << " " << get_error_string(result) \
              << std::endl;                                                           \
  }

#define CUDA_CHECK_AND_THROW(result)                       \
  if (result != cudaSuccess) {                             \
    std::cerr << __FILE__ << ": line " << __LINE__ << " "; \
    throw Error(result);                                   \
  }

#endif // CUDA_ERROR_HPP
