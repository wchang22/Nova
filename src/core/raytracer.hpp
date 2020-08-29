#ifndef RAYTRACER_HPP
#define RAYTRACER_HPP

#include <oidn.hpp>

#include "backend/accelerator.hpp"
#include "intersectables/intersectable_manager.hpp"
#include "kernel_types/area_light.hpp"
#include "kernel_types/scene_params.hpp"
#include "material/material_loader.hpp"
#include "material/sky_loader.hpp"
#include "scene/area_light.hpp"
#include "scene/ground_plane.hpp"

namespace nova {

class Scene;

class Raytracer {
public:
  Raytracer();

  void start();
  void set_scene(const Scene& scene);
  image_utils::image<uchar4> raytrace(bool denoise = false);

  void step() { sample_index++; }
  bool is_done() const { return sample_index >= num_samples; }
  int get_sample_index() const { return sample_index; }

private:
  IntersectableManager intersectable_manager;
  MaterialLoader material_loader;
  SkyLoader sky_loader;
  Accelerator accelerator;
  oidn::DeviceRef denoise_device;
  oidn::FilterRef denoise_filter;
  bool denoise_available;

  // Scene params and buffers
  std::string loaded_model;
  std::string loaded_sky;
  std::vector<AreaLight> loaded_lights;
  std::optional<GroundPlane> loaded_ground_plane;
  uint32_t width;
  uint32_t height;
  Image2DWrite<uchar4> color_img;
  Image2DArray<float4> temp_img1;
  Image2DArray<float4> temp_img2;
  Image2DArray<float4> prev_img;
  Wrapper<uint2> pixel_dims_wrapper;
  Wrapper<SceneParams> scene_params_wrapper;
  Buffer<TriangleData> triangle_buf;
  Buffer<TriangleMetaData> tri_meta_buf;
  Buffer<FlatBVHNode> bvh_buf;
  Buffer<AreaLightData> light_buf;
  Wrapper<uint32_t> num_lights_wrapper;
  Image2DArray<float4> material_imgs;
  Image2DRead<float4> sky_img;
  int num_samples = 0;

  // Local state
  int sample_index = 0;
};

}

#endif // RAYTRACER_HPP
