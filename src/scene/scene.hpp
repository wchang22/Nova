#include <glad/glad.h>

#include "camera/camera.hpp"
#include "core/raytracer.hpp"
#include "vector/vector_types.hpp"

namespace nova {

class Scene {
public:
  Scene();

  struct Settings {
    vec2i output_dimensions;
    std::string output_file_path;
    bool anti_aliasing;
    std::string model_path;
    std::string sky_path;
    Camera camera;
    vec3f light_position;
    vec3f light_intensity;
    vec3f shading_diffuse;
    float shading_metallic;
    float shading_roughness;
    float shading_ambient_occlusion;
    int ray_bounces;
    float exposure;

    bool operator==(const Settings& other) const {
      // clang-format off
      return output_dimensions == other.output_dimensions &&
             output_file_path == other.output_file_path &&
             anti_aliasing == other.anti_aliasing &&
             model_path == other.model_path &&
             sky_path == other.sky_path &&
             camera == other.camera &&
             light_position == other.light_position &&
             light_intensity == other.light_intensity &&
             shading_diffuse == other.shading_diffuse &&
             shading_metallic == other.shading_metallic &&
             shading_roughness == other.shading_roughness &&
             shading_ambient_occlusion == other.shading_ambient_occlusion &&
             ray_bounces == other.ray_bounces &&
             exposure == other.exposure;
      // clang-format on
    }
  };

  void init_texture();
  void cleanup_texture();

  const std::string& set_model_path(const std::string& path);
  const std::string& get_model_path() const;
  const std::string& set_sky_path(const std::string& path);
  const std::string& get_sky_path() const;
  bool set_anti_aliasing(bool anti_aliasing);
  bool get_anti_aliasing() const;
  vec3f set_camera_position(const vec3f& position);
  vec3f get_camera_position() const;
  vec3f set_camera_target(const vec3f& target);
  vec3f get_camera_target() const;
  vec3f set_camera_up(const vec3f& up);
  vec3f get_camera_up() const;
  float set_camera_fovy(float fovy);
  float get_camera_fovy() const;
  void move_camera(Camera::Direction direction, float speed);
  EyeCoords get_camera_eye_coords() const;
  const vec3f& set_light_position(const vec3f& position);
  const vec3f& get_light_position() const;
  const vec3f& set_light_intensity(const vec3f& intensity);
  const vec3f& get_light_intensity() const;
  const vec3f& set_shading_diffuse(const vec3f& diffuse);
  const vec3f& get_shading_diffuse() const;
  float set_shading_metallic(float metallic);
  float get_shading_metallic() const;
  float set_shading_roughness(float roughness);
  float get_shading_roughness() const;
  float set_shading_ambient_occlusion(float ambient_occlusion);
  float get_shading_ambient_occlusion() const;
  int set_ray_bounces(int bounces);
  int get_ray_bounces() const;
  float set_exposure(float exposure);
  float get_exposure() const;

  const vec2i& set_output_dimensions(const vec2i& dimensions);
  const vec2i& get_output_dimensions() const;
  const std::string& set_output_file_path(const std::string& path);
  const std::string& get_output_file_path() const;

  void render_to_screen();
  void render_to_image();
  GLuint get_scene_texture_id() const;

private:
  Settings settings;
  Settings prev_settings;
  Raytracer raytracer;
  GLuint scene_texture_id;
};

}