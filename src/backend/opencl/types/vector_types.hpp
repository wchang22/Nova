#ifndef OPENCL_VECTOR_TYPES_HPP
#define OPENCL_VECTOR_TYPES_HPP

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

template <typename vec_type>
inline auto& x(vec_type& f) { return f.s[0]; }
template <typename vec_type>
inline const auto& x(const vec_type& f) { return f.s[0]; }

template <typename vec_type>
inline auto& y(vec_type& f) { return f.s[1]; }
template <typename vec_type>
inline const auto& y(const vec_type& f) { return f.s[1]; }

template <typename vec_type>
inline auto& z(vec_type& f) { return f.s[2]; }
template <typename vec_type>
inline const auto& z(const vec_type& f) { return f.s[2]; }

template <typename vec_type>
inline auto& w(vec_type& f) { return f.s[3]; }
template <typename vec_type>
inline const auto& w(const vec_type& f) { return f.s[3]; }

#endif // OPENCL_VECTOR_TYPES_HPP