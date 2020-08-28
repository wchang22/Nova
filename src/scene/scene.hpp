#include <glad/glad.h>
#include <optional>

#include "camera/camera.hpp"
#include "core/raytracer.hpp"
#include "scene/area_light.hpp"
#include "scene/ground_plane.hpp"
#include "scene/scene_parser.hpp"
#include "vector/vector_types.hpp"

namespace nova {

class Scene {
public:
  Scene();

  struct Settings {
    vec2i output_dimensions;
    std::string output_file_path;
    bool path_tracing;
    int num_samples;
    bool last_frame_denoise;
    bool anti_aliasing;
    float exposure;
    std::string model_path;
    std::string sky_path;
    Camera camera;
    std::vector<AreaLight> lights;
    std::optional<GroundPlane> ground_plane;
    vec3f shading_diffuse;
    float shading_metallic;
    float shading_roughness;

    bool operator==(const Settings& other) const {
      // clang-format off
      return output_dimensions == other.output_dimensions &&
             output_file_path == other.output_file_path &&
             path_tracing == other.path_tracing &&
             num_samples == other.num_samples &&
             last_frame_denoise == other.last_frame_denoise &&
             anti_aliasing == other.anti_aliasing &&
             exposure == other.exposure &&
             model_path == other.model_path &&
             sky_path == other.sky_path &&
             camera == other.camera &&
             lights == other.lights &&
             ground_plane == other.ground_plane &&
             shading_diffuse == other.shading_diffuse &&
             shading_metallic == other.shading_metallic &&
             shading_roughness == other.shading_roughness;
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
  bool set_last_frame_denoise(bool last_frame_denoise);
  bool get_last_frame_denoise() const;
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
  void add_light();
  void delete_light(uint32_t index);
  vec3f set_light_position(uint32_t index, const vec3f& position);
  vec3f get_light_position(uint32_t index) const;
  vec3f set_light_normal(uint32_t index, const vec3f& normal);
  vec3f get_light_normal(uint32_t index) const;
  vec2f set_light_dims(uint32_t index, const vec2f& dims);
  vec2f get_light_dims(uint32_t index) const;
  vec3f set_light_intensity(uint32_t index, const vec3f& intensity);
  vec3f get_light_intensity(uint32_t index) const;
  const std::vector<AreaLight>& get_lights() const;
  void add_ground_plane();
  void delete_ground_plane();
  vec3f set_ground_plane_position(const vec3f& position);
  vec3f get_ground_plane_position() const;
  vec3f set_ground_plane_normal(const vec3f& normal);
  vec3f get_ground_plane_normal() const;
  vec2f set_ground_plane_dims(const vec2f& dims);
  vec2f get_ground_plane_dims() const;
  vec3f set_ground_plane_diffuse(const vec3f& diffuse);
  vec3f get_ground_plane_diffuse() const;
  float set_ground_plane_metallic(float metallic);
  float get_ground_plane_metallic() const;
  float set_ground_plane_roughness(float roughness);
  float get_ground_plane_roughness() const;
  const std::optional<GroundPlane>& get_ground_plane() const;
  const vec3f& set_shading_diffuse(const vec3f& diffuse);
  const vec3f& get_shading_diffuse() const;
  float set_shading_metallic(float metallic);
  float get_shading_metallic() const;
  float set_shading_roughness(float roughness);
  float get_shading_roughness() const;

  const vec2i& set_output_dimensions(const vec2i& dimensions);
  const vec2i& get_output_dimensions() const;
  const std::string& set_output_file_path(const std::string& path);
  const std::string& get_output_file_path() const;
  int set_num_samples(int num_samples);
  int get_num_samples() const;
  bool set_path_tracing(bool path_tracing);
  bool get_path_tracing() const;

  void render_to_screen();
  void render_to_image(bool single = false);
  GLuint get_scene_texture_id() const;

  int get_sample_index() const { return raytracer.get_sample_index(); };

private:
  Settings settings;
  Settings prev_settings;
  Raytracer raytracer;
  GLuint scene_texture_id;
};

}