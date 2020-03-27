#include "camera.hpp"

Camera::Camera(const glm::vec3& position,
               const glm::vec3& forward,
               const glm::vec3& up,
               uint32_t width,
               uint32_t height,
               float fovy)
  : position(position), forward(forward), up(up), width(width), height(height), fovy(fovy) {
  glm::vec3 normalized_forward = glm::normalize(forward);
  pitch = glm::degrees(std::asin(normalized_forward.y));
  yaw = glm::degrees(std::atan2(-normalized_forward.x, -normalized_forward.z));
}

void Camera::set_position(const glm::vec3& position) { this->position = position; }

void Camera::set_forward(const glm::vec3& forward) {
  this->forward = forward;
}

void Camera::set_up(const glm::vec3& up) { this->up = up; }

void Camera::set_width(uint32_t width) { this->width = width; }

void Camera::set_height(uint32_t height) { this->height = height; }

void Camera::set_fovy(float fovy) { this->fovy = fovy; }

const glm::vec3& Camera::get_position() const { return position; }

const glm::vec3& Camera::get_forward() const { return forward; }

const glm::vec3& Camera::get_up() const { return up; }

float Camera::get_fovy() const { return fovy; }

void Camera::update_direction(float delta_x, float delta_y) {
  yaw += delta_x;
  pitch = std::clamp(pitch + delta_y, -89.0f, 89.0f);

  forward.x = std::cos(glm::radians(pitch)) * std::cos(glm::radians(yaw));
  forward.y = std::sin(glm::radians(pitch));
  forward.z = std::cos(glm::radians(pitch)) * std::sin(glm::radians(yaw));
  forward = glm::normalize(forward);
}

void Camera::move(Direction direction, float speed) {
  switch (direction) {
    case Direction::FORWARD:
      position += glm::normalize(forward) * speed;
      break;
    case Direction::BACKWARD:
      position += -glm::normalize(forward) * speed;
      break;
    case Direction::LEFT:
      position += glm::normalize(glm::cross(glm::normalize(up), glm::normalize(forward))) * speed;
      break;
    case Direction::RIGHT:
      position += glm::normalize(glm::cross(glm::normalize(forward), glm::normalize(up))) * speed;
      break;
    case Direction::UP:
      position += glm::normalize(up) * speed;
      break;
    case Direction::DOWN:
      position += -glm::normalize(up) * speed;
      break;
  }
}

void Camera::zoom(float delta) {
  fovy = std::clamp(fovy + delta, 1.0f, 45.0f);
}

EyeCoords Camera::get_eye_coords() const {
  glm::vec2 half_fov(glm::vec2(fovy * width / height, fovy) / 2.0f);
  glm::vec2 coord_dims(glm::vec2(width, height) / 2.0f);
  glm::vec2 coord_scale(glm::tan(glm::radians(half_fov)) / coord_dims);

  glm::vec3 w = -glm::normalize(forward);
  glm::vec3 u = glm::normalize(glm::cross(up, w));
  glm::vec3 v = glm::cross(w, u);

  return {
    { coord_scale.x, coord_scale.y },
    { coord_dims.x, coord_dims.y },
    { position.x, position.y, position.z },
    { { u.x, u.y, u.z }, { v.x, v.y, v.z }, { w.x, w.y, w.z } },
  };
}
