#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#ifdef BACKEND_OPENCL
  #include "backend/opencl/types/vector_types.hpp"
#elif defined(BACKEND_CUDA)
  #include "backend/cuda/types/vector_types.hpp"
#endif

inline void align_dims(uint2& global_dims, const uint2& local_dims) {
  constexpr auto align_component = [](uint32_t& global_comp, const uint32_t& local_comp) {
    uint32_t r = global_comp % local_comp;
    global_comp += (r == 0) ? 0 : local_comp - r;
  };
  align_component(x(global_dims), x(local_dims));
  align_component(y(global_dims), y(local_dims));
}

#endif // COMMON_UTILS_HPP
