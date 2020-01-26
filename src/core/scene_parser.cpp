#include "scene_parser.h"
#include "constants.h"

SceneParser::SceneParser() : parsed_data(toml::parse(SCENE_PATH)) {}

std::string SceneParser::get_model_path() const {
  return ASSETS_PATH + toml::find<std::string>(parsed_data, "model", "path");
}

CameraSettings SceneParser::get_camera_settings() const {
  const auto position = toml::find<std::array<float, 3>>(parsed_data, "camera", "position");
  const auto forward = toml::find<std::array<float, 3>>(parsed_data, "camera", "forward");
  const auto up = toml::find<std::array<float, 3>>(parsed_data, "camera", "up");
  const auto fovy = toml::find<float>(parsed_data, "camera", "fovy");

  return {
    { position[0], position[1], position[2] },
    { forward[0], forward[1], forward[2] },
    { up[0], up[1], up[2] },
    fovy
  };
}

vec3 SceneParser::get_light_position() const {
  const auto position = toml::find<std::array<float, 3>>(parsed_data, "light", "position");
  return { position[0], position[1], position[2] };
}

ShadingDefaultSettings SceneParser::get_shading_default_settings() const {
  const auto ambient = toml::find<float>(parsed_data, "shading_defaults", "ambient");
  const auto diffuse = toml::find<float>(parsed_data, "shading_defaults", "diffuse");
  const auto specular = toml::find<float>(parsed_data, "shading_defaults", "specular");
  const auto shininess = toml::find<int>(parsed_data, "shading_defaults", "shininess");

  return {
    ambient,
    diffuse,
    specular,
    shininess
  };
}

unsigned int SceneParser::get_ray_recursion_depth() const {
  return toml::find<unsigned int>(parsed_data, "ray_recursion", "depth");
}