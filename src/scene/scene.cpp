#include "scene.hpp"
#include "constants.hpp"
#include "util/image/imageutils.hpp"
#include "util/profiling/profiling.hpp"
#include "vector/vector_conversions.hpp"

namespace nova {

Scene::Scene() {
  SceneParser scene_parser;

  const auto [output_dimensions, output_file_path, path_tracing, num_samples] =
    scene_parser.get_output_settings();
  const auto [model_paths, sky_path] = scene_parser.get_model_settings();
  const auto [last_frame_denoise, anti_aliasing, exposure] =
    scene_parser.get_post_processing_settings();
  const auto [camera_position, camera_forward, camera_up, camera_fovy] =
    scene_parser.get_camera_settings();
  const auto [parsed_lights] = scene_parser.get_light_settings();
  const auto [parsed_ground_plane] = scene_parser.get_ground_settings();
  const auto [default_diffuse, default_metallic, default_roughness] =
    scene_parser.get_shading_default_settings();

  Camera camera(vec_to_glm(camera_position), vec_to_glm(camera_forward), vec_to_glm(camera_up),
                { output_dimensions[0], output_dimensions[1] }, camera_fovy);

  std::vector<AreaLight> lights;
  std::transform(parsed_lights.begin(), parsed_lights.end(), std::back_inserter(lights),
                 [](const auto& light) -> AreaLight {
                   const auto& [light_intensity, light_position, light_normal, light_dims] = light;
                   return {
                     vec_to_glm(light_intensity),
                     vec_to_glm(light_position),
                     vec_to_glm(light_normal),
                     vec_to_glm(light_dims),
                   };
                 });

  std::optional<GroundPlane> ground_plane {};

  if (parsed_ground_plane.has_value()) {
    const auto& [gp_position, gp_normal, gp_dims, gp_diffuse, gp_metallic, gp_roughness] =
      parsed_ground_plane.value();
    ground_plane = {
      vec_to_glm(gp_position), vec_to_glm(gp_normal), vec_to_glm(gp_dims),
      vec_to_glm(gp_diffuse),  gp_metallic,           gp_roughness,
    };
  }

  settings = { output_dimensions,  output_file_path, path_tracing,     num_samples,
               last_frame_denoise, anti_aliasing,    exposure,         model_paths.front(),
               sky_path,           camera,           lights,           ground_plane,
               default_diffuse,    default_metallic, default_roughness };

  prev_settings.last_frame_denoise = last_frame_denoise;
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

bool Scene::set_last_frame_denoise(bool last_frame_denoise) {
  return settings.last_frame_denoise = last_frame_denoise;
}
bool Scene::get_last_frame_denoise() const { return settings.last_frame_denoise; }

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

void Scene::add_light() {
  if (settings.lights.empty()) {
    settings.lights.emplace_back();
  } else {
    settings.lights.emplace_back(settings.lights.back());
  }
}

void Scene::delete_light(uint32_t index) { settings.lights.erase(settings.lights.begin() + index); }

vec3f Scene::set_light_position(uint32_t index, const vec3f& position) {
  settings.lights[index].position = vec_to_glm(position);
  return position;
}

vec3f Scene::get_light_position(uint32_t index) const {
  return glm_to_vec(settings.lights[index].position);
}

vec3f Scene::set_light_normal(uint32_t index, const vec3f& normal) {
  settings.lights[index].normal = vec_to_glm(normal);
  return normal;
}

vec3f Scene::get_light_normal(uint32_t index) const {
  return glm_to_vec(settings.lights[index].normal);
}

vec2f Scene::set_light_dims(uint32_t index, const vec2f& dims) {
  settings.lights[index].dims = vec_to_glm(vec2f {
    std::max(dims[0], 0.0f),
    std::max(dims[1], 0.0f),
  });
  return glm_to_vec(settings.lights[index].dims);
}

vec2f Scene::get_light_dims(uint32_t index) const {
  return glm_to_vec(settings.lights[index].dims);
}

vec3f Scene::set_light_intensity(uint32_t index, const vec3f& intensity) {
  settings.lights[index].intensity = vec_to_glm(vec3f {
    std::max(intensity[0], 0.0f),
    std::max(intensity[1], 0.0f),
    std::max(intensity[2], 0.0f),
  });
  return glm_to_vec(settings.lights[index].intensity);
}

vec3f Scene::get_light_intensity(uint32_t index) const {
  return glm_to_vec(settings.lights[index].intensity);
}

const std::vector<AreaLight>& Scene::get_lights() const { return settings.lights; }

void Scene::add_ground_plane() {
  if (!settings.ground_plane.has_value()) {
    settings.ground_plane.emplace();
  }
}
void Scene::delete_ground_plane() { settings.ground_plane.reset(); }
vec3f Scene::set_ground_plane_position(const vec3f& position) {
  settings.ground_plane->position = vec_to_glm(position);
  return glm_to_vec(settings.ground_plane->position);
}
vec3f Scene::get_ground_plane_position() const {
  return glm_to_vec(settings.ground_plane->position);
  ;
}
vec3f Scene::set_ground_plane_normal(const vec3f& normal) {
  settings.ground_plane->normal = vec_to_glm(normal);
  return glm_to_vec(settings.ground_plane->normal);
}
vec3f Scene::get_ground_plane_normal() const { return glm_to_vec(settings.ground_plane->normal); }
vec2f Scene::set_ground_plane_dims(const vec2f& dims) {
  settings.ground_plane->dims = vec_to_glm(vec2f {
    std::max(dims[0], 0.0f),
    std::max(dims[1], 0.0f),
  });
  return glm_to_vec(settings.ground_plane->dims);
}
vec2f Scene::get_ground_plane_dims() const { return glm_to_vec(settings.ground_plane->dims); }
vec3f Scene::set_ground_plane_diffuse(const vec3f& diffuse) {
  settings.ground_plane->diffuse = vec_to_glm(vec3f {
    std::clamp(diffuse[0], 0.0f, 1.0f),
    std::clamp(diffuse[1], 0.0f, 1.0f),
    std::clamp(diffuse[2], 0.0f, 1.0f),
  });
  return glm_to_vec(settings.ground_plane->diffuse);
}
vec3f Scene::get_ground_plane_diffuse() const { return glm_to_vec(settings.ground_plane->diffuse); }
float Scene::set_ground_plane_metallic(float metallic) {
  return settings.ground_plane->metallic = std::clamp(metallic, 0.0f, 1.0f);
}
float Scene::get_ground_plane_metallic() const { return settings.ground_plane->metallic; }
float Scene::set_ground_plane_roughness(float roughness) {
  return settings.ground_plane->roughness = std::clamp(roughness, 0.0f, 1.0f);
}
float Scene::get_ground_plane_roughness() const { return settings.ground_plane->roughness; }
const std::optional<GroundPlane>& Scene::get_ground_plane() const { return settings.ground_plane; }

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

bool Scene::set_path_tracing(bool path_tracing) { return settings.path_tracing = path_tracing; }

bool Scene::get_path_tracing() const { return settings.path_tracing; }

int Scene::set_num_samples(int num_samples) {
  return settings.num_samples = std::max(1, num_samples);
}

int Scene::get_num_samples() const { return settings.num_samples; }

void Scene::render_to_screen() {
  bool needs_denoise = settings.last_frame_denoise && !prev_settings.last_frame_denoise;

  // If nothing has changed, no need to update
  if (settings != prev_settings) {
    raytracer.set_scene(*this);

    // Hack: If only `num_samples` or `last_frame_denoise` changed, don't restart
    Settings temp_settings = settings;
    temp_settings.num_samples = prev_settings.num_samples;
    temp_settings.last_frame_denoise = prev_settings.last_frame_denoise;

    if (temp_settings != prev_settings) {
      raytracer.start();
    }
    prev_settings = settings;
  }
  if (raytracer.is_done() && !needs_denoise) {
    return;
  }

  needs_denoise |=
    settings.last_frame_denoise && raytracer.get_sample_index() == settings.num_samples - 1;
  image_utils::image<uchar4> im = raytracer.raytrace(needs_denoise);
  raytracer.step();

  // Bind image data to OpenGL texture for rendering
  glBindTexture(GL_TEXTURE_2D, scene_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im.width, im.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               im.data.data());
  glBindTexture(GL_TEXTURE_2D, 0);
}

void Scene::render_to_image(bool single) {
  PROFILE_SCOPE("Render to Image");

  image_utils::image<uchar4> img;

  if (!single || raytracer.get_sample_index() == 0) {
    raytracer.set_scene(*this);
    raytracer.start();
  }
  PROFILE_SECTION_START("Profile Loop");
  if (single) {
    img = raytracer.raytrace(settings.last_frame_denoise);
  } else {
    while (!raytracer.is_done()) {
      img = raytracer.raytrace(false);
      raytracer.step();
    }
    if (settings.last_frame_denoise) {
      img = raytracer.raytrace(true);
    }
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write Image");
  image_utils::write_image(settings.output_file_path.c_str(), img);
  PROFILE_SECTION_END();
}

GLuint Scene::get_scene_texture_id() const { return scene_texture_id; }

}