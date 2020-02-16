#include "camera.h"
#include <iostream>
#include <glm/gtx/string_cast.hpp>

Camera::Camera(const vec3& position, const vec3& forward,
               const vec3& up, uint32_t width, uint32_t height, float fovy)
  : position(position),
    forward(forward),
    up(up),
    width(width),
    height(height),
    fovy(fovy)
{
}

EyeCoords Camera::get_eye_coords() const {
  vec2 half_fov(vec2(fovy * width / height, fovy) / 2.0f);
  vec2 coord_dims(vec2(width, height) / 2.0f);
  vec2 coord_scale(tan(radians(half_fov)) / coord_dims);

  vec3 w = -normalize(forward);
  vec3 u = normalize(cross(up, w));
  vec3 v = cross(w, u);

  return {
    { {coord_scale.x, coord_scale.y} },
    { {coord_dims.x, coord_dims.y} },
    { {position.x, position.y, position.z} },
    { {u.x, u.y, u.z} }, { {v.x, v.y, v.z} }, { {w.x, w.y, w.z} }
  };
}
