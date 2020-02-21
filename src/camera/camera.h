#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

#include "kernel_types/eye_coords.h"

using namespace glm;

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
