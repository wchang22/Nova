#ifndef SCENE_PARSER_HPP
#define SCENE_PARSER_HPP

#include <toml.hpp>

#include "kernel_types/area_light.hpp"
#include "vector/vector_types.hpp"

namespace nova {

struct OutputSettings {
  vec2i dimensions;
  std::string file_path;
  int num_samples;
};

struct ModelSettings {
  std::vector<std::string> model_paths;
  std::string sky_path;
};

struct PostProcessingSettings {
  bool anti_aliasing;
  float exposure;
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
};

struct LightSettings {
  vec3f intensity;
  vec3f position;
  vec3f normal;
  float size;
};

class SceneParser {
public:
  SceneParser();

  OutputSettings get_output_settings() const;
  ModelSettings get_model_settings() const;
  PostProcessingSettings get_post_processing_settings() const;
  CameraSettings get_camera_settings() const;
  LightSettings get_light_settings() const;
  ShadingDefaultSettings get_shading_default_settings() const;

private:
  toml::value parsed_data;
};

}

#endif // SCENE_PARSER_HPP