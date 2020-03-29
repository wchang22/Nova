#ifndef KERNEL_TYPE_EYE_COORDS_HPP
#define KERNEL_TYPE_EYE_COORDS_HPP

#include "backend/types.hpp"
#include "kernel_types/matrix.hpp"

namespace nova {

struct EyeCoords {
  float2 coord_scale;
  float2 coord_dims;
  float3 eye_pos;
  Mat3x3 eye_coord_frame;
};

}

#endif // KERNEL_TYPE_EYE_COORDS_HPP
