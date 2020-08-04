#include "scene_parser.hpp"
#include "constants.hpp"

namespace nova {

SceneParser::SceneParser() : parsed_data(toml::parse(SCENE_PATH)) {}

OutputSettings SceneParser::get_output_settings() const {
  vec2i dimensions = toml::find<vec2i>(parsed_data, "output", "dimensions");
  std::string file_path = toml::find<std::string>(parsed_data, "output", "file_path");
  int num_samples = toml::find<int>(parsed_data, "output", "num_samples");

  return { dimensions, file_path, num_samples };
}

PostProcessingSettings SceneParser::get_post_processing_settings() const {
  bool anti_aliasing = toml::find<bool>(parsed_data, "post_processing", "anti_aliasing");
  float exposure = toml::find<float>(parsed_data, "post_processing", "exposure");

  return { anti_aliasing, exposure };
}

ModelSettings SceneParser::get_model_settings() const {
  std::vector<std::string> model_paths =
    toml::find<std::vector<std::string>>(parsed_data, "model", "paths");
  for (auto& path : model_paths) {
    path.insert(0, ASSETS_PATH);
  }
  std::string sky_path = ASSETS_PATH + toml::find<std::string>(parsed_data, "model", "sky");

  return { model_paths, sky_path };
}

CameraSettings SceneParser::get_camera_settings() const {
  vec3f position = toml::find<vec3f>(parsed_data, "camera", "position");
  vec3f target = toml::find<vec3f>(parsed_data, "camera", "target");
  vec3f up = toml::find<vec3f>(parsed_data, "camera", "up");
  float fovy = toml::find<float>(parsed_data, "camera", "fovy");

  return { position, target, up, fovy };
}

LightSettings SceneParser::get_light_settings() const {
  vec3f intensity = toml::find<vec3f>(parsed_data, "light", "intensity");
  vec3f position = toml::find<vec3f>(parsed_data, "light", "position");
  vec3f normal = toml::find<vec3f>(parsed_data, "light", "normal");
  float size = toml::find<float>(parsed_data, "light", "size");

  return { intensity, position, normal, size };
}

ShadingDefaultSettings SceneParser::get_shading_default_settings() const {
  vec3f diffuse = toml::find<vec3f>(parsed_data, "shading_defaults", "diffuse");
  float metallic = toml::find<float>(parsed_data, "shading_defaults", "metallic");
  float roughness = toml::find<float>(parsed_data, "shading_defaults", "roughness");

  return { diffuse, metallic, roughness };
}

}