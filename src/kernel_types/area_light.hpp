#ifndef KERNEL_TYPE_AREA_LIGHT_HPP
#define KERNEL_TYPE_AREA_LIGHT_HPP

#include "backend/types.hpp"

namespace nova {

struct AreaLight {
  float3 intensity;
  float3 position;
  float3 normal;
  float size;
};

}

#endif // KERNEL_TYPE_AREA_LIGHT_HPP
