#include "scene_parser.hpp"
#include "constants.hpp"

namespace nova {

SceneParser::SceneParser() : parsed_data(toml::parse(SCENE_PATH)) {}

OutputSettings SceneParser::get_output_settings() const {
  vec2i dimensions = toml::find<vec2i>(parsed_data, "output", "dimensions");
  std::string file_path = toml::find<std::string>(parsed_data, "output", "file_path");

  return { dimensions, file_path };
}

std::vector<std::string> SceneParser::get_model_paths() const {
  std::vector<std::string> paths = toml::find<std::vector<std::string>>(parsed_data, "model", "paths");
  for (auto& path : paths) {
    path.insert(0, ASSETS_PATH);
  }

  return paths;
}

CameraSettings SceneParser::get_camera_settings() const {
  vec3f position = toml::find<vec3f>(parsed_data, "camera", "position");
  vec3f target = toml::find<vec3f>(parsed_data, "camera", "target");
  vec3f up = toml::find<vec3f>(parsed_data, "camera", "up");
  float fovy = toml::find<float>(parsed_data, "camera", "fovy");

  return { position, target, up, fovy };
}

LightSettings SceneParser::get_light_settings() const {
  vec3f position = toml::find<vec3f>(parsed_data, "light", "position");
  vec3f intensity = toml::find<vec3f>(parsed_data, "light", "intensity");

  return { position, intensity };
}

ShadingDefaultSettings SceneParser::get_shading_default_settings() const {
  vec3f diffuse = toml::find<vec3f>(parsed_data, "shading_defaults", "diffuse");
  float metallic = toml::find<float>(parsed_data, "shading_defaults", "metallic");
  float roughness = toml::find<float>(parsed_data, "shading_defaults", "roughness");
  float ambient_occlusion = toml::find<float>(parsed_data, "shading_defaults", "ambient_occlusion");

  return { diffuse, metallic, roughness, ambient_occlusion };
}

int SceneParser::get_ray_bounces() const {
  return toml::find<int>(parsed_data, "ray_bounces", "number");
}

}