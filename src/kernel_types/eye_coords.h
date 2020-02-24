#ifndef KERNEL_TYPE_EYE_COORDS_H
#define KERNEL_TYPE_EYE_COORDS_H

#include "backend/types.h"
#include "kernel_types/matrix.h"

struct EyeCoords {
  float2 coord_scale;
  float2 coord_dims;
  float3 eye_pos;
  Mat3x3 eye_coord_frame;
};

#endif // KERNEL_TYPE_EYE_COORDS_H
