#include <array>
#include <glad/glad.h>

#include "core/raytracer.hpp"

class Scene {
public:
  Scene();

  struct Settings {
    std::string model_path;
    std::array<float, 3> camera_position;
    std::array<float, 3> camera_forward;
    std::array<float, 3> camera_up;
    float camera_fovy;
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
  const std::array<float, 3>& set_camera_position(const std::array<float, 3>& position);
  const std::array<float, 3>& get_camera_position() const;
  const std::array<float, 3>& set_camera_forward(const std::array<float, 3>& forward);
  const std::array<float, 3>& get_camera_forward() const;
  const std::array<float, 3>& set_camera_up(const std::array<float, 3>& up);
  const std::array<float, 3>& get_camera_up() const;
  float set_camera_fovy(float fovy);
  float get_camera_fovy() const;
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

  void set_width(uint32_t width);
  uint32_t get_width() const;
  void set_height(uint32_t height);
  uint32_t get_height() const;

  void render();
  GLuint get_scene_texture_id() const;

private:
  uint32_t width;
  uint32_t height;
  Settings settings;
  Raytracer raytracer;
  GLuint scene_texture_id;
};