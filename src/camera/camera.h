#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

#include "backend/types.h"

using namespace glm;

struct EyeCoords {
  float2 coord_scale;
  float2 coord_dims;
  float3 eye_pos;
  float3 eye_coord_frame0;
  float3 eye_coord_frame1;
  float3 eye_coord_frame2;
};

class Camera {
public:
  Camera(const vec3& position, const vec3& forward, const vec3& up,
         uint32_t width, uint32_t height, float fovy);

  EyeCoords get_eye_coords() const;

private:
  vec3 position;
  vec3 forward;
  vec3 up;

  uint32_t width, height;
  float fovy;
};

#endif // CAMERA_H
