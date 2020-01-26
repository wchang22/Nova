#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include <toml.hpp>
#include <glm/glm.hpp>

using namespace glm;

struct CameraSettings {
  vec3 position;
  vec3 forward;
  vec3 up;
  float fovy;
};

struct ShadingDefaultSettings {
  float ambient;
  float diffuse;
  float specular;
  int shininess;
};

class SceneParser {
public:
  SceneParser();

  std::string get_model_path() const;
  CameraSettings get_camera_settings() const;
  vec3 get_light_position() const;
  ShadingDefaultSettings get_shading_default_settings() const;
  unsigned int get_ray_recursion_depth() const;

private:
  toml::value parsed_data;
};

#endif // SCENE_PARSER_H