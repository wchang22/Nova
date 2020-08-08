#ifndef KERNEL_TYPE_AREA_LIGHT_HPP
#define KERNEL_TYPE_AREA_LIGHT_HPP

#include "backend/types.hpp"

namespace nova {

struct AreaLightData {
  float3 intensity;
  float3 position;
  float3 normal;
  float2 dims;
};

}

#endif // KERNEL_TYPE_AREA_LIGHT_HPP
