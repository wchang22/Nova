#ifndef OPENCL_VECTOR_TYPES_H
#define OPENCL_VECTOR_TYPES_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

using float2 = cl_float2;
using float3 = cl_float3;
using float4 = cl_float4;

template <typename floatn>
inline float& x(floatn& f) { return f.s[0]; }
template <typename floatn>
inline const float& x(const floatn& f) { return f.s[0]; }

template <typename floatn>
inline float& y(floatn& f) { return f.s[1]; }
template <typename floatn>
inline const float& y(const floatn& f) { return f.s[1]; }

template <typename floatn>
inline float& z(floatn& f) { return f.s[2]; }
template <typename floatn>
inline const float& z(const floatn& f) { return f.s[2]; }

template <typename floatn>
inline float& w(floatn& f) { return f.s[3]; }
template <typename floatn>
inline const float& w(const floatn& f) { return f.s[3]; }

#endif // OPENCL_VECTOR_TYPES_H