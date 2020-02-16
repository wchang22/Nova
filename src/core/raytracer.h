#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "scene_parser.h"
#include "camera/camera.h"
#include "intersectables/intersectable_manager.h"
#include "material/material_loader.h"
#include "backend/accelerator.h"

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

#endif // RAYTRACER_H
