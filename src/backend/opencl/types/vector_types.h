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
using char2 = cl_char2;
using char3 = cl_char3;
using char4 = cl_char4;
using uchar2 = cl_uchar2;
using uchar3 = cl_uchar3;
using uchar4 = cl_uchar4;
using int2 = cl_int2;
using int3 = cl_int3;
using int4 = cl_int4;
using uint2 = cl_uint2;
using uint3 = cl_uint3;
using uint4 = cl_uint4;

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