#ifndef BVH_H
#define BVH_H

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <limits>
#include <fstream>

#include "intersectables/triangle.h"
#include "intersectables/aabb.h"
#include "kernel_types/bvh_node.h"

using namespace glm;

const vec3 VEC_MAX(std::numeric_limits<float>::max());

struct BVHNode {
  AABB aabb;
  std::vector<Triangle> triangles;
  std::unique_ptr<BVHNode> left;
  std::unique_ptr<BVHNode> right;

  BVHNode() : aabb({ -VEC_MAX, VEC_MAX }) {}
  float get_cost() const { return aabb.get_cost(triangles.size()); }
};

std::istream& operator>>(std::istream& in, FlatBVHNode& node);
std::ostream& operator<<(std::ostream& out, const FlatBVHNode& node);

class BVH {
public:
  BVH(const std::string& name, std::vector<Triangle>& triangles);

  // Note: Modifies `triangles`
  std::vector<FlatBVHNode> build();
private:
  std::unique_ptr<BVHNode> build_bvh();
  std::vector<FlatBVHNode> build_flat_bvh(std::unique_ptr<BVHNode>& root);
  void build_bvh_node(std::unique_ptr<BVHNode>& node, int depth, const float root_sa);
  size_t build_flat_bvh_vec(std::vector<FlatBVHNode>& flat_nodes, std::unique_ptr<BVHNode>& node);

  std::string name;
  std::vector<Triangle>& triangles;
};

#endif
