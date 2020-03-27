#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glm/glm.hpp>

#include "kernel_types/eye_coords.hpp"

class Camera {
public:
  Camera() = default;
  Camera(const glm::vec3& position,
         const glm::vec3& target,
         const glm::vec3& up,
         uint32_t width,
         uint32_t height,
         float fovy);

  enum class Direction {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN,
  };

  void set_position(const glm::vec3& position);
  void set_target(const glm::vec3& target);
  void set_up(const glm::vec3& up);
  void set_width(uint32_t width);
  void set_height(uint32_t height);
  void set_fovy(float fovy);

  const glm::vec3& get_position() const;
  const glm::vec3& get_target() const;
  const glm::vec3& get_up() const;
  float get_fovy() const;

  void move(Direction direction, float speed);

  EyeCoords get_eye_coords() const;

private:
  glm::vec3 position;
  glm::vec3 target;
  glm::vec3 up;

  uint32_t width, height;
  float fovy;
  float pitch;
  float yaw;
};

#endif // CAMERA_HPP
