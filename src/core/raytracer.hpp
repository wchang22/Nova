#ifndef RAYTRACER_HPP
#define RAYTRACER_HPP

#include "backend/accelerator.hpp"
#include "intersectables/intersectable_manager.hpp"
#include "kernel_types/scene_params.hpp"
#include "material/material_loader.hpp"
#include "material/sky_loader.hpp"

namespace nova {

class Scene;

class Raytracer {
public:
  Raytracer();

  void start();
  void set_scene(const Scene& scene);
  image_utils::image<uchar4> raytrace();

  void step() { sample_index++; }
  bool is_done() const { return sample_index >= num_samples; }
  int get_sample_index() const { return sample_index; }

private:
  IntersectableManager intersectable_manager;
  MaterialLoader material_loader;
  SkyLoader sky_loader;
  Accelerator accelerator;

  // Scene params and buffers
  std::string loaded_model;
  std::string loaded_sky;
  uint32_t width;
  uint32_t height;
  Image2DWrite<uchar4> pixel_im;
  Image2DRead<float4> prev_pixel_im;
  Image2DReadWrite<float4> temp_pixel_im1;
  Image2DReadWrite<float4> temp_pixel_im2;
  Wrapper<uint2> pixel_dims_wrapper;
  Wrapper<SceneParams> scene_params_wrapper;
  Buffer<TriangleData> triangle_buf;
  Buffer<TriangleMetaData> tri_meta_buf;
  Buffer<FlatBVHNode> bvh_buf;
  Buffer<int2> rem_coords_buf;
  Buffer<uint32_t> rem_pixels_buf;
  Image2DArray<float4> material_ims;
  Image2DRead<float4> sky_im;
  int num_samples = 0;

  // Local state
  int sample_index = 0;
};

}

#endif // RAYTRACER_HPP
