#include "scene.hpp"
#include "constants.hpp"
#include "scene/scene_parser.hpp"
#include "util/image/imageutils.hpp"
#include "util/profiling/profiling.hpp"
#include "vector/vector_conversions.hpp"

namespace nova {

Scene::Scene() {
  SceneParser scene_parser;

  const auto [output_dimensions, output_file_path, num_samples] =
    scene_parser.get_output_settings();
  const auto [model_paths, sky_path] = scene_parser.get_model_settings();
  const auto [anti_aliasing, exposure] = scene_parser.get_post_processing_settings();
  const auto [camera_position, camera_forward, camera_up, camera_fovy] =
    scene_parser.get_camera_settings();
  const auto [light_intensity, light_position, light_normal, light_size] =
    scene_parser.get_light_settings();
  const auto [default_diffuse, default_metallic, default_roughness] =
    scene_parser.get_shading_default_settings();

  Camera camera(vec_to_glm(camera_position), vec_to_glm(camera_forward), vec_to_glm(camera_up),
                { output_dimensions[0], output_dimensions[1] }, camera_fovy);

  settings = {
    output_dimensions, output_file_path,    num_samples,       anti_aliasing,
    exposure,          model_paths.front(), sky_path,          camera,
    light_intensity,   light_position,      light_normal,      light_size,
    default_diffuse,   default_metallic,    default_roughness
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

const std::string& Scene::set_sky_path(const std::string& path) { return settings.sky_path = path; }

const std::string& Scene::get_sky_path() const { return settings.sky_path; }

vec3f Scene::set_camera_position(const vec3f& position) {
  settings.camera.set_position(vec_to_glm(position));
  return position;
}

bool Scene::set_anti_aliasing(bool anti_aliasing) { return settings.anti_aliasing = anti_aliasing; }

bool Scene::get_anti_aliasing() const { return settings.anti_aliasing; }

float Scene::set_exposure(float exposure) { return settings.exposure = std::max(exposure, 0.01f); }

float Scene::get_exposure() const { return settings.exposure; }

vec3f Scene::get_camera_position() const { return glm_to_vec(settings.camera.get_position()); }

vec3f Scene::set_camera_target(const vec3f& target) {
  settings.camera.set_target(vec_to_glm(target));
  return target;
}

vec3f Scene::get_camera_target() const { return glm_to_vec(settings.camera.get_target()); }

vec3f Scene::set_camera_up(const vec3f& up) {
  settings.camera.set_up(vec_to_glm(up));
  return up;
}

vec3f Scene::get_camera_up() const { return glm_to_vec(settings.camera.get_up()); }

float Scene::set_camera_fovy(float fovy) {
  float clamped_fovy = std::clamp(fovy, 1.0f, 45.0f);
  settings.camera.set_fovy(clamped_fovy);
  return clamped_fovy;
}

float Scene::get_camera_fovy() const { return settings.camera.get_fovy(); }

void Scene::move_camera(Camera::Direction direction, float speed) {
  settings.camera.move(direction, speed);
}

EyeCoords Scene::get_camera_eye_coords() const { return settings.camera.get_eye_coords(); }

const vec3f& Scene::set_light_position(const vec3f& position) {
  return settings.light_position = position;
}

const vec3f& Scene::get_light_position() const { return settings.light_position; }

const vec3f& Scene::set_light_normal(const vec3f& normal) { return settings.light_normal = normal; }

const vec3f& Scene::get_light_normal() const { return settings.light_normal; }

float Scene::set_light_size(float size) { return settings.light_size = std::max(0.0f, size); }

float Scene::get_light_size() const { return settings.light_size; }

const vec3f& Scene::set_light_intensity(const vec3f& intensity) {
  return settings.light_intensity = {
    std::max(intensity[0], 0.0f),
    std::max(intensity[1], 0.0f),
    std::max(intensity[2], 0.0f),
  };
}

const vec3f& Scene::get_light_intensity() const { return settings.light_intensity; }

const vec3f& Scene::set_shading_diffuse(const vec3f& diffuse) {
  return settings.shading_diffuse = {
    std::clamp(diffuse[0], 0.0f, 1.0f),
    std::clamp(diffuse[1], 0.0f, 1.0f),
    std::clamp(diffuse[2], 0.0f, 1.0f),
  };
}

const vec3f& Scene::get_shading_diffuse() const { return settings.shading_diffuse; }

float Scene::set_shading_metallic(float metallic) {
  return settings.shading_metallic = std::clamp(metallic, 0.0f, 1.0f);
}

float Scene::get_shading_metallic() const { return settings.shading_metallic; }

float Scene::set_shading_roughness(float roughness) {
  return settings.shading_roughness = std::clamp(roughness, 0.0f, 1.0f);
}

float Scene::get_shading_roughness() const { return settings.shading_roughness; }

const vec2i& Scene::set_output_dimensions(const vec2i& dimensions) {
  settings.output_dimensions[0] = std::clamp(dimensions[0], 1, MAX_RESOLUTION.first);
  settings.output_dimensions[1] = std::clamp(dimensions[1], 1, MAX_RESOLUTION.second);
  settings.camera.set_dimensions({ settings.output_dimensions[0], settings.output_dimensions[1] });
  return settings.output_dimensions;
}

const vec2i& Scene::get_output_dimensions() const { return settings.output_dimensions; }

const std::string& Scene::set_output_file_path(const std::string& path) {
  return settings.output_file_path = path;
}

const std::string& Scene::get_output_file_path() const { return settings.output_file_path; }

int Scene::set_num_samples(int num_samples) {
  return settings.num_samples = std::max(1, num_samples);
}

int Scene::get_num_samples() const { return settings.num_samples; }

void Scene::render_to_screen() {
  // If nothing has changed, no need to update
  if (settings != prev_settings) {
    raytracer.set_scene(*this);

    // Hack: If only `num_samples` changed, don't restart
    Settings temp_settings = settings;
    temp_settings.num_samples = prev_settings.num_samples;

    if (temp_settings != prev_settings) {
      raytracer.start();
    }
    prev_settings = settings;
  }
  if (raytracer.is_done()) {
    return;
  }

  image_utils::image<uchar4> im = raytracer.raytrace();
  raytracer.step();

  // Bind image data to OpenGL texture for rendering
  glBindTexture(GL_TEXTURE_2D, scene_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im.width, im.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               im.data.data());
  glBindTexture(GL_TEXTURE_2D, 0);
}

void Scene::render_to_image(bool single) {
  PROFILE_SCOPE("Render to Image");

  image_utils::image<uchar4> im;

  if (!single || raytracer.get_sample_index() == 0) {
    raytracer.set_scene(*this);
    raytracer.start();
  }
  PROFILE_SECTION_START("Profile Loop");
  if (single) {
    im = raytracer.raytrace();
  } else {
    while (!raytracer.is_done()) {
      im = raytracer.raytrace();
      raytracer.step();
    }
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write Image");
  image_utils::write_image(settings.output_file_path.c_str(), im);
  PROFILE_SECTION_END();
}

GLuint Scene::get_scene_texture_id() const { return scene_texture_id; }

}