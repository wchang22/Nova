#ifndef RAYTRACER_HPP
#define RAYTRACER_HPP

#include <unordered_set>

#include "backend/accelerator.hpp"
#include "intersectables/intersectable_manager.hpp"
#include "material/material_loader.hpp"
#include "scene/scene_parser.hpp"

class Scene;

class Raytracer {
public:
  Raytracer();

  image_utils::image raytrace(const Scene& scene, uint32_t width, uint32_t height);

private:
  SceneParser scene_parser;
  IntersectableManager intersectable_manager;
  MaterialLoader material_loader;
  Accelerator accelerator;
  std::unordered_set<std::string> loaded_models;
};

#endif // RAYTRACER_HPP
