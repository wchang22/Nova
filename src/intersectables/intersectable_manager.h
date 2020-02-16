#ifndef INTERSECTABLE_MANAGER_H
#define INTERSECTABLE_MANAGER_H

#include <vector>
#include <unordered_map>

#include "model/model.h"
#include "intersectables/triangle.h"
#include "bvh/bvh.h"

struct IntersectableData {
  std::vector<TriangleData> triangle_data;
  std::vector<TriangleMetaData> triangle_meta_data;
  std::vector<FlatBVHNode> bvh_data;
};

class IntersectableManager {
public:
  IntersectableManager(const std::string& name);
  void add_triangle(const Triangle& tri, const TriangleMeta& meta);
  void add_model(const Model& model);
  IntersectableData build();

private:
  std::string name;
  std::vector<Triangle> triangles;
  std::unordered_map<Triangle, TriangleMeta, TriangleHash> triangle_map;
};

#endif // INTERSECTABLE_MANAGER_H
