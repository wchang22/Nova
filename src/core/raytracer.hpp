#ifndef RAYTRACER_HPP
#define RAYTRACER_HPP

#include "backend/accelerator.hpp"
#include "intersectables/intersectable_manager.hpp"
#include "kernel_types/eye_coords.hpp"
#include "material/material_loader.hpp"
#include "scene/scene_parser.hpp"

class Scene;

class Raytracer {
public:
  Raytracer();

  void set_scene(const Scene& scene, uint32_t width, uint32_t height);
  image_utils::image raytrace();

private:
  SceneParser scene_parser;
  IntersectableManager intersectable_manager;
  MaterialLoader material_loader;
  Accelerator accelerator;

  // Scene params and buffers
  std::string loaded_model;
  uint32_t width;
  uint32_t height;
  Buffer<uchar4> pixel_buf;
  Wrapper<uint2> pixel_dims_wrapper;
  Wrapper<EyeCoords> ec;
  Buffer<TriangleData> triangle_buf;
  Buffer<TriangleMetaData> tri_meta_buf;
  Buffer<FlatBVHNode> bvh_buf;
  Buffer<uint2> rem_coords_buf;
  Buffer<uint32_t> rem_pixels_buf;
  Image2DArray<uchar4> material_ims;
};

#endif // RAYTRACER_HPP
