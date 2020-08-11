#ifndef KERNEL_TYPE_AREA_LIGHT_HPP
#define KERNEL_TYPE_AREA_LIGHT_HPP

#include "backend/types.hpp"

namespace nova {

enum class AreaLightType : bool {
  RECT = 0,
  TRI = 1,
};

struct AreaLightRect {
  float3 position;
  float3 normal;
  float3 dims;
};

struct AreaLightTri {
  float3 v1;
  float3 v2;
  float3 v3;
};

struct AreaLightData {
  float3 intensity;
  union {
    AreaLightRect rect;
    AreaLightTri tri;
  };
  AreaLightType type;
};

}

#endif // KERNEL_TYPE_AREA_LIGHT_HPP
