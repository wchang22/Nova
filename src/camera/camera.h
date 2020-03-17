#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

#include "kernel_types/eye_coords.h"

class Camera {
public:
  Camera(const glm::vec3& position, const glm::vec3& forward, const glm::vec3& up,
         uint32_t width, uint32_t height, float fovy);

  EyeCoords get_eye_coords() const;

private:
  glm::vec3 position;
  glm::vec3 forward;
  glm::vec3 up;

  uint32_t width, height;
  float fovy;
};

#endif // CAMERA_H
