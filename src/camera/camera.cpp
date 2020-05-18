#include <glm/gtx/transform.hpp>

#include "camera.hpp"
#include "vector/vector_conversions.hpp"

namespace nova {

Camera::Camera(const glm::vec3& position,
               const glm::vec3& target,
               const glm::vec3& up,
               const std::pair<uint32_t, uint32_t>& dimensions,
               float fovy)
  : position(position), target(target), up(up), dimensions(dimensions), fovy(fovy) {}

bool Camera::operator==(const Camera& other) const {
  // clang-format off
  return position == other.position && 
         target == other.target && 
         up == other.up &&
         dimensions == other.dimensions && 
         fovy == other.fovy;
  // clang-format on
}

void Camera::set_position(const glm::vec3& position) { this->position = position; }

void Camera::set_target(const glm::vec3& target) { this->target = target; }

void Camera::set_up(const glm::vec3& up) { this->up = up; }

void Camera::set_dimensions(const std::pair<uint32_t, uint32_t>& dimensions) {
  this->dimensions = dimensions;
}

void Camera::set_fovy(float fovy) { this->fovy = fovy; }

const glm::vec3& Camera::get_position() const { return position; }

const glm::vec3& Camera::get_target() const { return target; }

const glm::vec3& Camera::get_up() const { return up; }

float Camera::get_fovy() const { return fovy; }

void Camera::move(Direction direction, float speed) {
  constexpr float rotate_threshold = 0.98f;
  constexpr float pan_multiplier = 0.025f;
  constexpr float forward_multiplier = 0.5f;

  glm::vec3 w = -glm::normalize(target - position);
  glm::vec3 u = glm::normalize(glm::cross(up, w));
  glm::vec3 v = glm::cross(w, u);

  glm::vec3 forward = position - target;
  float distance = glm::length(forward);

  // Slow down movement as we move close to the target position
  constexpr auto dist_mod = [](float x) {
    return std::exp(2.0f * std::min(x, 2.334f) - 4.0f) + 0.05f;
  };

  switch (direction) {
    case Direction::FORWARD:
      // Prevent camera position from moving past target position
      if (distance >= 0.1f || speed < 0.0f) {
        position += -w * glm::sign(speed) * dist_mod(distance) * forward_multiplier;
      }
      break;
    case Direction::RIGHT:
      position += pan_multiplier * speed * u;
      target += pan_multiplier * speed * u;
      break;
    case Direction::UP:
      position += pan_multiplier * speed * v;
      target += pan_multiplier * speed * v;
      break;
    case Direction::ROTATE_RIGHT:
      position = glm::mat3(glm::rotate(glm::radians(speed), v)) * forward + target;
      break;
    case Direction::ROTATE_UP:
      // Prevent camera from moving past overhead/underneath position
      if (glm::dot(w, glm::normalize(up)) * glm::sign(speed) < rotate_threshold) {
        position = glm::mat3(glm::rotate(-glm::radians(speed), u)) * forward + target;
      }
      break;
  }
}

EyeCoords Camera::get_eye_coords() const {
  const auto& [width, height] = dimensions;
  glm::vec2 half_fov(glm::vec2(fovy * width / height, fovy) * 0.5f);
  glm::vec2 coord_dims(glm::vec2(width, height) * 0.5f);
  glm::vec2 coord_scale(glm::tan(glm::radians(half_fov)) / coord_dims);

  glm::vec3 w = -glm::normalize(target - position);
  glm::vec3 u = glm::normalize(glm::cross(up, w));
  glm::vec3 v = glm::cross(w, u);

  return {
    glm_to_float2(coord_scale),
    glm_to_float2(coord_dims),
    glm_to_float3(position),
    { glm_to_float3(u), glm_to_float3(v), glm_to_float3(w) },
  };
}

}
