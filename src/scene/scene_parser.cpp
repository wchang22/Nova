#include <algorithm>

#include "constants.hpp"
#include "scene_parser.hpp"

namespace nova {

SceneParser::SceneParser() : parsed_data(toml::parse(SCENE_PATH)) {}

OutputSettings SceneParser::get_output_settings() const {
  vec2i dimensions = toml::find<vec2i>(parsed_data, "output", "dimensions");
  std::string file_path = toml::find<std::string>(parsed_data, "output", "file_path");
  int num_samples = toml::find<int>(parsed_data, "output", "num_samples");

  return { dimensions, file_path, num_samples };
}

PostProcessingSettings SceneParser::get_post_processing_settings() const {
  bool last_frame_denoise = toml::find<bool>(parsed_data, "post_processing", "last_frame_denoise");
  bool anti_aliasing = toml::find<bool>(parsed_data, "post_processing", "anti_aliasing");
  float exposure = toml::find<float>(parsed_data, "post_processing", "exposure");

  return { last_frame_denoise, anti_aliasing, exposure };
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
  try {
    std::vector<toml::table> lights_table =
      toml::find<std::vector<toml::table>>(parsed_data, "light");
    std::vector<ParsedLight> lights;

    std::transform(
      lights_table.begin(), lights_table.end(), std::back_inserter(lights),
      [](const auto& table) -> ParsedLight {
        return { toml::get<vec3f>(table.at("intensity")), toml::get<vec3f>(table.at("position")),
                 toml::get<vec3f>(table.at("normal")), toml::get<vec2f>(table.at("dims")) };
      });

    return { lights };
  } catch (const std::out_of_range& error) {
    return {};
  }
}

ShadingDefaultSettings SceneParser::get_shading_default_settings() const {
  vec3f diffuse = toml::find<vec3f>(parsed_data, "shading_defaults", "diffuse");
  float metallic = toml::find<float>(parsed_data, "shading_defaults", "metallic");
  float roughness = toml::find<float>(parsed_data, "shading_defaults", "roughness");

  return { diffuse, metallic, roughness };
}

GroundSettings SceneParser::get_ground_settings() const {
  try {
    toml::table table = toml::find<toml::table>(parsed_data, "ground");
    return { ParsedGroundPlane {
      toml::get<vec3f>(table.at("position")), toml::get<vec3f>(table.at("normal")),
      toml::get<vec2f>(table.at("dims")), toml::get<vec3f>(table.at("diffuse")),
      toml::get<float>(table.at("metallic")), toml::get<float>(table.at("roughness")) } };
  } catch (const std::out_of_range& error) {
    return {};
  }
}

}