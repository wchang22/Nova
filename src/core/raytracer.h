#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <CL/cl2.hpp>

using Kernel = cl::KernelFunctor<cl::Image2D>;

class Raytracer {
public:
  Raytracer(uint32_t width, uint32_t height);

  void raytrace();

private:
  uint32_t width, height;
  std::vector<uint8_t> image_buf;

  cl::Context context;
  cl::Device device;
  cl::CommandQueue queue;
  cl::Program program;
  cl::Image2D image;
  std::unique_ptr<Kernel> kernel;
};

#endif // RAYTRACER_H