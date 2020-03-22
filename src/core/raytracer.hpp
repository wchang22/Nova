#ifndef RAYTRACER_HPP
#define RAYTRACER_HPP

#include "backend/accelerator.hpp"
#include "camera/camera.hpp"
#include "intersectables/intersectable_manager.hpp"
#include "material/material_loader.hpp"
#include "scene_parser.hpp"

class Raytracer {
public:
  Raytracer(uint32_t width, uint32_t height, const std::string& name);

  void raytrace();

private:
  uint32_t width, height;
  std::string name;
  SceneParser scene_parser;
  CameraSettings camera_settings;
  Camera camera;
  IntersectableManager intersectable_manager;
  MaterialLoader material_loader;
  Accelerator accelerator;
};

#endif // RAYTRACER_HPP
