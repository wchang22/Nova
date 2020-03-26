#include "scene_parser.hpp"
#include "constants.hpp"

SceneParser::SceneParser() : parsed_data(toml::parse(SCENE_PATH)) {}

std::vector<std::string> SceneParser::get_model_paths() const {
  auto paths = toml::find<std::vector<std::string>>(parsed_data, "model", "paths");
  for (auto& path : paths) {
    path.insert(0, ASSETS_PATH);
  }

  return paths;
}

CameraSettings SceneParser::get_camera_settings() const {
  const auto position = toml::find<std::array<float, 3>>(parsed_data, "camera", "position");
  const auto forward = toml::find<std::array<float, 3>>(parsed_data, "camera", "forward");
  const auto up = toml::find<std::array<float, 3>>(parsed_data, "camera", "up");
  const auto fovy = toml::find<float>(parsed_data, "camera", "fovy");

  return { position, forward, up, fovy };
}

LightSettings SceneParser::get_light_settings() const {
  const auto position = toml::find<std::array<float, 3>>(parsed_data, "light", "position");
  const auto intensity = toml::find<std::array<float, 3>>(parsed_data, "light", "intensity");

  return {
    position,
    intensity,
  };
}

ShadingDefaultSettings SceneParser::get_shading_default_settings() const {
  const auto diffuse = toml::find<std::array<float, 3>>(parsed_data, "shading_defaults", "diffuse");
  const auto metallic = toml::find<float>(parsed_data, "shading_defaults", "metallic");
  const auto roughness = toml::find<float>(parsed_data, "shading_defaults", "roughness");
  const auto ambient_occlusion =
    toml::find<float>(parsed_data, "shading_defaults", "ambient_occlusion");

  return { diffuse, metallic, roughness, ambient_occlusion };
}

int SceneParser::get_ray_recursion_depth() const {
  return toml::find<int>(parsed_data, "ray_recursion", "depth");
}