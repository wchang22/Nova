#include "error.h"

Error::Error(cudaError_t error)
  : std::runtime_error(cudaGetErrorName(error)),
    error(error) {}

cudaError_t Error::err() const {
  return error;
}