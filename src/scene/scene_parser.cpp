#include "scene_parser.hpp"
#include "constants.hpp"

SceneParser::SceneParser() : parsed_data(toml::parse(SCENE_PATH)) {}

OutputSettings SceneParser::get_output_settings() const {
  auto dimensions = toml::find<std::array<int, 2>>(parsed_data, "output", "dimensions");
  auto file_path = toml::find<std::string>(parsed_data, "output", "file_path");

  return { dimensions, file_path };
}

std::vector<std::string> SceneParser::get_model_paths() const {
  auto paths = toml::find<std::vector<std::string>>(parsed_data, "model", "paths");
  for (auto& path : paths) {
    path.insert(0, ASSETS_PATH);
  }

  return paths;
}

CameraSettings SceneParser::get_camera_settings() const {
  auto position = toml::find<std::array<float, 3>>(parsed_data, "camera", "position");
  auto target = toml::find<std::array<float, 3>>(parsed_data, "camera", "target");
  auto up = toml::find<std::array<float, 3>>(parsed_data, "camera", "up");
  auto fovy = toml::find<float>(parsed_data, "camera", "fovy");

  return { position, target, up, fovy };
}

LightSettings SceneParser::get_light_settings() const {
  auto position = toml::find<std::array<float, 3>>(parsed_data, "light", "position");
  auto intensity = toml::find<std::array<float, 3>>(parsed_data, "light", "intensity");

  return { position, intensity };
}

ShadingDefaultSettings SceneParser::get_shading_default_settings() const {
  auto diffuse = toml::find<std::array<float, 3>>(parsed_data, "shading_defaults", "diffuse");
  auto metallic = toml::find<float>(parsed_data, "shading_defaults", "metallic");
  auto roughness = toml::find<float>(parsed_data, "shading_defaults", "roughness");
  auto ambient_occlusion = toml::find<float>(parsed_data, "shading_defaults", "ambient_occlusion");

  return { diffuse, metallic, roughness, ambient_occlusion };
}

int SceneParser::get_ray_bounces() const {
  return toml::find<int>(parsed_data, "ray_bounces", "number");
}