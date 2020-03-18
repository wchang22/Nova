#ifndef SCENE_PARSER_HPP
#define SCENE_PARSER_HPP

#include <toml.hpp>
#include <glm/glm.hpp>

struct CameraSettings {
  glm::vec3 position;
  glm::vec3 forward;
  glm::vec3 up;
  float fovy;
};

struct ShadingDefaultSettings {
  glm::vec3 diffuse;
  float metallic;
  float roughness;
  float ambient_occlusion;
};

struct LightSettings {
  glm::vec3 position;
  glm::vec3 intensity;
};

class SceneParser {
public:
  SceneParser();

  std::vector<std::string> get_model_paths() const;
  CameraSettings get_camera_settings() const;
  LightSettings get_light_settings() const;
  ShadingDefaultSettings get_shading_default_settings() const;
  int get_ray_recursion_depth() const;

private:
  toml::value parsed_data;
};

#endif // SCENE_PARSER_HPP