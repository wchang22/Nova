#ifndef INTERSECTABLE_MANAGER_HPP
#define INTERSECTABLE_MANAGER_HPP

#include <unordered_map>
#include <vector>

#include "bvh/bvh.hpp"
#include "intersectables/triangle.hpp"
#include "kernel_types/bvh_node.hpp"
#include "model/model.hpp"

struct IntersectableData {
  std::vector<TriangleData> triangle_data;
  std::vector<TriangleMetaData> triangle_meta_data;
  std::vector<FlatBVHNode> bvh_data;
};

class IntersectableManager {
public:
  void add_triangle(const Triangle& tri, const TriangleMeta& meta);
  void add_model(const Model& model);
  void clear();
  IntersectableData build();

private:
  std::vector<Triangle> triangles;
  std::unordered_map<Triangle, TriangleMeta, TriangleHash> triangle_map;
};

#endif // INTERSECTABLE_MANAGER_HPP
