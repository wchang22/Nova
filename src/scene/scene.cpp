#include "scene.hpp"
#include "scene/scene_parser.hpp"

Scene::Scene() {
  SceneParser scene_parser;

  const auto model_paths = scene_parser.get_model_paths();
  const auto [camera_position, camera_forward, camera_up, camera_fovy] =
    scene_parser.get_camera_settings();
  const auto [light_position, light_intensity] = scene_parser.get_light_settings();
  const int ray_bounces = scene_parser.get_ray_bounces();
  const auto [default_diffuse, default_metallic, default_roughness, default_ambient_occlusion] =
    scene_parser.get_shading_default_settings();

  settings = {
    model_paths.front(), camera_position,  camera_forward,    camera_up,
    camera_fovy,         light_position,   light_intensity,   ray_bounces,
    default_diffuse,     default_metallic, default_roughness, default_ambient_occlusion
  };
}

void Scene::init_texture() {
  glGenTextures(1, &scene_texture_id);
  glBindTexture(GL_TEXTURE_2D, scene_texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void Scene::cleanup_texture() { glDeleteTextures(1, &scene_texture_id); }

const std::string& Scene::set_model_path(const std::string& path) {
  return settings.model_path = path;
}

const std::string& Scene::get_model_path() const { return settings.model_path; }

const std::array<float, 3>& Scene::set_camera_position(const std::array<float, 3>& position) {
  return settings.camera_position = position;
}

const std::array<float, 3>& Scene::get_camera_position() const { return settings.camera_position; }

const std::array<float, 3>& Scene::set_camera_forward(const std::array<float, 3>& forward) {
  return settings.camera_forward = forward;
}

const std::array<float, 3>& Scene::get_camera_forward() const { return settings.camera_forward; }

const std::array<float, 3>& Scene::set_camera_up(const std::array<float, 3>& up) {
  return settings.camera_up = up;
}

const std::array<float, 3>& Scene::get_camera_up() const { return settings.camera_up; }

float Scene::set_camera_fovy(float fovy) {
  return settings.camera_fovy = std::clamp(fovy, 1.0f, 45.0f);
}

float Scene::get_camera_fovy() const { return settings.camera_fovy; }

const std::array<float, 3>& Scene::set_light_position(const std::array<float, 3>& position) {
  return settings.light_position = position;
}

const std::array<float, 3>& Scene::get_light_position() const { return settings.light_position; }

const std::array<float, 3>& Scene::set_light_intensity(const std::array<float, 3>& intensity) {
  return settings.light_intensity = {
    std::max(intensity[0], 0.0f),
    std::max(intensity[1], 0.0f),
    std::max(intensity[2], 0.0f),
  };
}

const std::array<float, 3>& Scene::get_light_intensity() const { return settings.light_intensity; }

int Scene::set_ray_bounces(int bounces) { return settings.ray_bounces = std::max(1, bounces); }

int Scene::get_ray_bounces() const { return settings.ray_bounces; }

const std::array<float, 3>& Scene::set_shading_diffuse(const std::array<float, 3>& diffuse) {
  return settings.shading_diffuse = {
    std::clamp(diffuse[0], 0.0f, 1.0f),
    std::clamp(diffuse[1], 0.0f, 1.0f),
    std::clamp(diffuse[2], 0.0f, 1.0f),
  };
}

const std::array<float, 3>& Scene::get_shading_diffuse() const { return settings.shading_diffuse; }

float Scene::set_shading_metallic(float metallic) {
  return settings.shading_metallic = std::clamp(metallic, 0.0f, 1.0f);
}

float Scene::get_shading_metallic() const { return settings.shading_metallic; }

float Scene::set_shading_roughness(float roughness) {
  return settings.shading_roughness = std::clamp(roughness, 0.0f, 1.0f);
}

float Scene::get_shading_roughness() const { return settings.shading_roughness; }

float Scene::set_shading_ambient_occlusion(float ambient_occlusion) {
  return settings.shading_ambient_occlusion = std::clamp(ambient_occlusion, 0.0f, 1.0f);
}

float Scene::get_shading_ambient_occlusion() const { return settings.shading_ambient_occlusion; }

void Scene::set_width(uint32_t width) { this->width = width; }

uint32_t Scene::get_width() const { return width; }

void Scene::set_height(uint32_t height) { this->height = height; }

uint32_t Scene::get_height() const { return height; }

void Scene::render() {
  raytracer.set_scene(*this, width, height);
  auto im = raytracer.raytrace();
  glBindTexture(GL_TEXTURE_2D, scene_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im.width, im.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               im.data.data());
  glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint Scene::get_scene_texture_id() const { return scene_texture_id; }