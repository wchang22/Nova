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

  Camera camera({ camera_position[0], camera_position[1], camera_position[2] },
                { camera_forward[0], camera_forward[1], camera_forward[2] },
                { camera_up[0], camera_up[1], camera_up[2] }, width, height, camera_fovy);

  settings = {
    model_paths.front(), camera,           light_position,    light_intensity,          ray_bounces,
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

std::array<float, 3> Scene::set_camera_position(const std::array<float, 3>& position) {
  settings.camera.set_position({ position[0], position[1], position[2] });
  return position;
}

std::array<float, 3> Scene::get_camera_position() const {
  const glm::vec3& position = settings.camera.get_position();
  return { position.x, position.y, position.z };
}

std::array<float, 3> Scene::set_camera_target(const std::array<float, 3>& target) {
  settings.camera.set_target({ target[0], target[1], target[2] });
  return target;
}

std::array<float, 3> Scene::get_camera_target() const {
  const glm::vec3& target = settings.camera.get_target();
  return { target.x, target.y, target.z };
}

std::array<float, 3> Scene::set_camera_up(const std::array<float, 3>& up) {
  settings.camera.set_up({ up[0], up[1], up[2] });
  return up;
}

std::array<float, 3> Scene::get_camera_up() const {
  const glm::vec3& up = settings.camera.get_up();
  return { up.x, up.y, up.z };
}

float Scene::set_camera_fovy(float fovy) {
  float clamped_fovy = std::clamp(fovy, 1.0f, 45.0f);
  settings.camera.set_fovy(fovy);
  return clamped_fovy;
}

float Scene::get_camera_fovy() const { return settings.camera.get_fovy(); }

void Scene::move_camera(Camera::Direction direction, float speed) {
  settings.camera.move(direction, speed);
}

EyeCoords Scene::get_camera_eye_coords() const { return settings.camera.get_eye_coords(); }

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

void Scene::set_width(uint32_t width) {
  settings.camera.set_width(width);
  this->width = width;
}

uint32_t Scene::get_width() const { return width; }

void Scene::set_height(uint32_t height) {
  settings.camera.set_height(height);
  this->height = height;
}

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