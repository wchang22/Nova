#ifndef KERNEL_TYPE_WAVEFRONT_HPP
#define KERNEL_TYPE_WAVEFRONT_HPP

#include "backend/types.hpp"

namespace nova {

struct PackedRay {
  float4 origin_path_index;
  float3 direction;
};

struct Path {
  float3 throughput;
  float3 color;
  float3 albedo;
  float3 normal;
  int direct;
};

struct IntersectionData {
  float3 barycentric;
  float length;
  int tri_index;
  int ray_index;
};

}

#endif // KERNEL_TYPE_WAVEFRONT_HPP
