#ifndef INTERSECTABLE_MANAGER_HPP
#define INTERSECTABLE_MANAGER_HPP

#include <unordered_map>
#include <vector>

#include "bvh/bvh.hpp"
#include "intersectables/triangle.hpp"
#include "kernel_types/area_light.hpp"
#include "kernel_types/bvh_node.hpp"
#include "light/area_light.hpp"
#include "model/model.hpp"

namespace nova {

struct IntersectableData {
  std::vector<TriangleData> triangle_data;
  std::vector<TriangleMetaData> triangle_meta_data;
  std::vector<FlatBVHNode> bvh_data;
  std::vector<AreaLightData> light_data;
};

class IntersectableManager {
public:
  void add_triangle(const Triangle& tri, const TriangleMeta& meta);
  void add_model(const Model& model);
  void add_light(const AreaLight& light);
  void clear();
  IntersectableData build();

private:
  std::vector<Triangle> triangles;
  std::unordered_map<Triangle, TriangleMeta, TriangleHash> triangle_map;
  std::vector<AreaLight> lights;
  std::unordered_map<Triangle, uint32_t, TriangleHash> triangle_light_map;
};

}

#endif // INTERSECTABLE_MANAGER_HPP
