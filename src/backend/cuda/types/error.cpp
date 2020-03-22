#include "error.hpp"

Error::Error(cudaError_t error) : std::runtime_error(cudaGetErrorName(error)), error(error) {}

cudaError_t Error::err() const { return error; }

const char* get_error_string(cudaError_t code) { return cudaGetErrorString(code); }