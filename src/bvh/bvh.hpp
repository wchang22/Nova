#ifndef BVH_HPP
#define BVH_HPP

#include <fstream>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <vector>

#include "intersectables/aabb.hpp"
#include "intersectables/triangle.hpp"
#include "kernel_types/bvh_node.hpp"

struct BVHNode {
  AABB aabb;
  std::vector<Triangle> triangles;
  std::unique_ptr<BVHNode> left;
  std::unique_ptr<BVHNode> right;

  BVHNode() : aabb(AABB::make_no_intersection()) {}
  float get_cost() { return aabb.get_cost(triangles.size()); }
};

std::istream& operator>>(std::istream& in, FlatBVHNode& node);
std::ostream& operator<<(std::ostream& out, const FlatBVHNode& node);

class BVH {
public:
  BVH(const std::string& name, std::vector<Triangle>& triangles);

  // Note: Modifies `triangles`
  std::vector<FlatBVHNode> build();

private:
  struct SplitParams {
    float cost;
    uint32_t split_index;
    int axis;
    AABB left;
    AABB right;
    size_t left_num_triangles;
    size_t right_num_triangles;

    static SplitParams make_default() {
      return { std::numeric_limits<float>::max(), 0, -1, {}, {}, 0, 0 };
    }

    SplitParams min(SplitParams&& other) {
      return cost < other.cost ? std::move(*this) : std::move(other);
    }
  };

  struct Bin {
    AABB bounds;
    size_t num_triangles;
  };

  std::unique_ptr<BVHNode> build_bvh();
  void build_bvh_node(std::unique_ptr<BVHNode>& node, const int depth);

  std::vector<FlatBVHNode> build_flat_bvh(std::unique_ptr<BVHNode>& root);
  size_t build_flat_bvh_vec(std::vector<FlatBVHNode>& flat_nodes, std::unique_ptr<BVHNode>& node);

  SplitParams
  find_object_split(const std::unique_ptr<BVHNode>& node, int axis, const std::vector<Bin>& bins);
  std::pair<std::unique_ptr<BVHNode>, std::unique_ptr<BVHNode>>
  split_node(std::unique_ptr<BVHNode>& node,
             SplitParams&& best_params,
             const std::vector<std::pair<AABB, glm::uvec3>>& bound_centers);

  std::string name;
  std::vector<Triangle>& triangles;
};

#endif
