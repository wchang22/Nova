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
    int num_samples;
    bool anti_aliasing;
    float exposure;
    std::string model_path;
    std::string sky_path;
    Camera camera;
    vec3f light_intensity;
    vec3f light_position;
    vec3f light_normal;
    float light_size;
    vec3f shading_diffuse;
    float shading_metallic;
    float shading_roughness;
    float shading_ambient_occlusion;

    bool operator==(const Settings& other) const {
      // clang-format off
      return output_dimensions == other.output_dimensions &&
             output_file_path == other.output_file_path &&
             num_samples == other.num_samples &&
             anti_aliasing == other.anti_aliasing &&
             exposure == other.exposure &&
             model_path == other.model_path &&
             sky_path == other.sky_path &&
             camera == other.camera &&
             light_intensity == other.light_intensity &&
             light_position == other.light_position &&
             light_normal == other.light_normal &&
             light_size == other.light_size &&
             shading_diffuse == other.shading_diffuse &&
             shading_metallic == other.shading_metallic &&
             shading_roughness == other.shading_roughness &&
             shading_ambient_occlusion == other.shading_ambient_occlusion;
      // clang-format on
    }
    bool operator!=(const Settings& other) const { return !(*this == other); }
  };

  void init_texture();
  void cleanup_texture();

  const std::string& set_model_path(const std::string& path);
  const std::string& get_model_path() const;
  const std::string& set_sky_path(const std::string& path);
  const std::string& get_sky_path() const;
  bool set_anti_aliasing(bool anti_aliasing);
  bool get_anti_aliasing() const;
  float set_exposure(float exposure);
  float get_exposure() const;
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
  const vec3f& set_light_normal(const vec3f& normal);
  const vec3f& get_light_normal() const;
  float set_light_size(float size);
  float get_light_size() const;
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

  const vec2i& set_output_dimensions(const vec2i& dimensions);
  const vec2i& get_output_dimensions() const;
  const std::string& set_output_file_path(const std::string& path);
  const std::string& get_output_file_path() const;
  int set_num_samples(int num_samples);
  int get_num_samples() const;

  void render_to_screen();
  void render_to_image(bool single = true);
  GLuint get_scene_texture_id() const;

  int get_sample_index() const { return raytracer.get_sample_index(); };

private:
  Settings settings;
  Settings prev_settings;
  Raytracer raytracer;
  GLuint scene_texture_id;
};

}