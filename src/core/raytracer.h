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
  Raytracer(uint32_t width, uint32_t height);

  void raytrace();

private:
  SceneParser scene_parser;
  uint32_t width, height;
  CameraSettings camera_settings;
  Camera camera;
  std::string model_name;
  IntersectableManager intersectables;
  MaterialLoader material_loader;
  Model model;

  cl::Context context;
  cl::Device device;
  cl::CommandQueue queue;
  cl::Program program;
  cl::Image2D image;
  cl::Kernel kernel;
};

#endif // RAYTRACER_H
