#include <glad/glad.h>

#include "camera/camera.hpp"
#include "core/raytracer.hpp"

class Scene {
public:
  Scene();

  struct Settings {
    std::array<int, 2> output_dimensions;
    std::string output_file_path;
    std::string model_path;
    Camera camera;
    std::array<float, 3> light_position;
    std::array<float, 3> light_intensity;
    int ray_bounces;
    std::array<float, 3> shading_diffuse;
    float shading_metallic;
    float shading_roughness;
    float shading_ambient_occlusion;
  };

  void init_texture();
  void cleanup_texture();

  const std::string& set_model_path(const std::string& path);
  const std::string& get_model_path() const;
  std::array<float, 3> set_camera_position(const std::array<float, 3>& position);
  std::array<float, 3> get_camera_position() const;
  std::array<float, 3> set_camera_target(const std::array<float, 3>& target);
  std::array<float, 3> get_camera_target() const;
  std::array<float, 3> set_camera_up(const std::array<float, 3>& up);
  std::array<float, 3> get_camera_up() const;
  float set_camera_fovy(float fovy);
  float get_camera_fovy() const;
  void move_camera(Camera::Direction direction, float speed);
  EyeCoords get_camera_eye_coords() const;
  const std::array<float, 3>& set_light_position(const std::array<float, 3>& position);
  const std::array<float, 3>& get_light_position() const;
  const std::array<float, 3>& set_light_intensity(const std::array<float, 3>& intensity);
  const std::array<float, 3>& get_light_intensity() const;
  int set_ray_bounces(int bounces);
  int get_ray_bounces() const;
  const std::array<float, 3>& set_shading_diffuse(const std::array<float, 3>& diffuse);
  const std::array<float, 3>& get_shading_diffuse() const;
  float set_shading_metallic(float metallic);
  float get_shading_metallic() const;
  float set_shading_roughness(float roughness);
  float get_shading_roughness() const;
  float set_shading_ambient_occlusion(float ambient_occlusion);
  float get_shading_ambient_occlusion() const;

  const std::array<int, 2>& set_output_dimensions(const std::array<int, 2>& dimensions);
  const std::array<int, 2>& get_output_dimensions() const;
  const std::string& set_output_file_path(const std::string& path);
  const std::string& get_output_file_path() const;

  void render_to_screen();
  void render_to_image();
  GLuint get_scene_texture_id() const;

private:
  Settings settings;
  Raytracer raytracer;
  GLuint scene_texture_id;
};