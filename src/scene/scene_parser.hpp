#ifndef SCENE_PARSER_HPP
#define SCENE_PARSER_HPP

#include <toml.hpp>

#include "vector/vector_types.hpp"

namespace nova {

struct OutputSettings {
  vec2i dimensions;
  std::string file_path;
};

struct CameraSettings {
  vec3f position;
  vec3f target;
  vec3f up;
  float fovy;
};

struct ShadingDefaultSettings {
  vec3f diffuse;
  float metallic;
  float roughness;
  float ambient_occlusion;
};

struct LightSettings {
  vec3f position;
  vec3f intensity;
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

}

#endif // SCENE_PARSER_HPP