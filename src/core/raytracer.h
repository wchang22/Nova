#ifndef RAYTRACER_H
#define RAYTRACER_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

#include "scene_parser.h"
#include "camera/camera.h"
#include "intersectables/intersectable_manager.h"
#include "model/model.h"
#include "material/material_loader.h"

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

  cl::Context context;
  cl::CommandQueue queue;
  cl::Kernel kernel;
};

#endif // RAYTRACER_H
