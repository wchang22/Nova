#ifndef KERNELS_BACKEND_OPENCL_ASSERTION_HPP
#define KERNELS_BACKEND_OPENCL_ASSERTION_HPP

#ifdef NDEBUG
  #define assert(expr)
#else
  #define assert(expr) \
    (void) ((expr) ||  \
            (printf("%s:%d: %s: Assertion %s failed.\n", __FILE__, __LINE__, __func__, #expr), 0))
#endif

#endif