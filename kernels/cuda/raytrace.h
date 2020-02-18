#ifndef CUDA_KERNEL_RAYTRACE_H
#define CUDA_KERNEL_RAYTRACE_H

#include <tuple>

void raytrace(std::tuple<uint32_t, uint32_t, uint32_t> global_size,
              std::tuple<uint32_t, uint32_t, uint32_t> local_size);

#endif // CUDA_KERNEL_RAYTRACE_H