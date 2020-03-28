#ifndef SCENE_PARSER_HPP
#define SCENE_PARSER_HPP

#include <array>
#include <toml.hpp>

struct OutputSettings {
  std::array<int, 2> dimensions;
  std::string file_path;
};

struct CameraSettings {
  std::array<float, 3> position;
  std::array<float, 3> target;
  std::array<float, 3> up;
  float fovy;
};

struct ShadingDefaultSettings {
  std::array<float, 3> diffuse;
  float metallic;
  float roughness;
  float ambient_occlusion;
};

struct LightSettings {
  std::array<float, 3> position;
  std::array<float, 3> intensity;
};

class SceneParser {
public:
  SceneParser();

  OutputSettings get_output_settings() const;
  std::vector<std::string> get_model_paths() const;
  CameraSettings get_camera_settings() const;
  LightSettings get_light_settings() const;
  ShadingDefaultSettings get_shading_default_settings() const;
  int get_ray_bounces() const;

private:
  toml::value parsed_data;
};

#endif // SCENE_PARSER_HPP