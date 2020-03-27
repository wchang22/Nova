#include <glm/gtx/transform.hpp>

#include "camera.hpp"

Camera::Camera(const glm::vec3& position,
               const glm::vec3& target,
               const glm::vec3& up,
               uint32_t width,
               uint32_t height,
               float fovy)
  : position(position), target(target), up(up), width(width), height(height), fovy(fovy) {}

void Camera::set_position(const glm::vec3& position) { this->position = position; }

void Camera::set_target(const glm::vec3& target) { this->target = target; }

void Camera::set_up(const glm::vec3& up) { this->up = up; }

void Camera::set_width(uint32_t width) { this->width = width; }

void Camera::set_height(uint32_t height) { this->height = height; }

void Camera::set_fovy(float fovy) { this->fovy = fovy; }

const glm::vec3& Camera::get_position() const { return position; }

const glm::vec3& Camera::get_target() const { return target; }

const glm::vec3& Camera::get_up() const { return up; }

float Camera::get_fovy() const { return fovy; }

void Camera::move(Direction direction, float speed) {
  glm::vec3 w = -glm::normalize(target - position);
  glm::vec3 u = glm::normalize(glm::cross(up, w));
  glm::vec3 v = glm::cross(w, u);

  glm::vec3 forward = position - target;

  switch (direction) {
    case Direction::FORWARD:
      if (speed <= glm::length(forward)) {
        position += -w * speed;
      }
      break;
    case Direction::BACKWARD:
      position += w * speed;
      break;
    case Direction::LEFT:
      position = glm::mat3(glm::rotate(-glm::radians(speed), v)) * forward + target;
      break;
    case Direction::RIGHT:
      position = glm::mat3(glm::rotate(glm::radians(speed), v)) * forward + target;
      break;
    case Direction::UP:
      if (glm::dot(w, glm::normalize(up)) < 0.98f) {
        position = glm::mat3(glm::rotate(-glm::radians(speed), u)) * forward + target;
      }
      break;
    case Direction::DOWN:
      if (glm::dot(-w, glm::normalize(up)) < 0.98f) {
        position = glm::mat3(glm::rotate(glm::radians(speed), u)) * forward + target;
      }
      break;
  }
}

EyeCoords Camera::get_eye_coords() const {
  glm::vec2 half_fov(glm::vec2(fovy * width / height, fovy) / 2.0f);
  glm::vec2 coord_dims(glm::vec2(width, height) / 2.0f);
  glm::vec2 coord_scale(glm::tan(glm::radians(half_fov)) / coord_dims);

  glm::vec3 w = -glm::normalize(target - position);
  glm::vec3 u = glm::normalize(glm::cross(up, w));
  glm::vec3 v = glm::cross(w, u);

  return {
    { coord_scale.x, coord_scale.y },
    { coord_dims.x, coord_dims.y },
    { position.x, position.y, position.z },
    { { u.x, u.y, u.z }, { v.x, v.y, v.z }, { w.x, w.y, w.z } },
  };
}
