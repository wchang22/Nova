#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <CL/cl2.hpp>

#include "camera/camera.h"
#include "intersectables/intersectable_manager.h"
#include "model/model.h"

using Kernel = cl::KernelFunctor<cl::Image2D, Camera::EyeCoords, cl::Buffer, int>;

class Raytracer {
public:
  Raytracer(uint32_t width, uint32_t height);

  void raytrace();

private:
  uint32_t width, height;
  std::vector<uint8_t> image_buf;
  Camera camera;
  IntersectableManager intersectables;
  Model model;

  cl::Context context;
  cl::Device device;
  cl::CommandQueue queue;
  cl::Program program;
  cl::Image2D image;
  std::unique_ptr<Kernel> kernel;
};

#endif // RAYTRACER_H