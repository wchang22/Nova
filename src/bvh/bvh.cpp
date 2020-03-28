#include <algorithm>
#include <cassert>
#include <filesystem>

#include "bvh.hpp"
#include "constants.hpp"
#include "util/exception/exception.hpp"
#include "util/profiling/profiling.hpp"
#include "util/serialization/serialization.hpp"

// Algorithm from https://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

BVH::BVH(std::vector<Triangle>& triangles) : triangles(triangles) {}

std::vector<FlatBVHNode> BVH::build() {
  PROFILE_SCOPE("Build BVH");

  PROFILE_SECTION_START("Build BVH Tree");
  std::unique_ptr<BVHNode> bvh = build_bvh();
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Build Flat BVH");
  std::vector<FlatBVHNode> flat_bvh = build_flat_bvh(bvh);
  PROFILE_SECTION_END();

  return flat_bvh;
}

std::unique_ptr<BVHNode> BVH::build_bvh() {
  auto root = std::make_unique<BVHNode>();

  root->triangles.reserve(triangles.size());

  // Create bounding box to capture all triangles
  for (const auto& triangle : triangles) {
    root->aabb.grow(triangle.get_bounds());
    root->triangles.emplace_back(std::move(triangle));
  }
  // Clear triangles, as they have all been moved out
  triangles.clear();

// Recursively build bvh in parallel
#pragma omp parallel
  {
#pragma omp single
    build_bvh_node(root, 0);
#pragma omp taskwait
  }
  return root;
}

BVH::SplitParams BVH::find_object_split(const std::unique_ptr<BVHNode>& node,
                                        int axis,
                                        const std::vector<Bin>& bins) {
  uint32_t num_bins = bins.size();
  uint32_t num_splits = num_bins - 1;
  uint32_t num_triangles = node->triangles.size();
  SplitParams best_params = SplitParams::make_default();

  // Build right split bounds incrementally from right to left
  std::vector<AABB> right_bounds(num_bins);
  right_bounds.back() = bins.back().bounds;
  for (int32_t i = num_bins - 2; i >= 0; i--) {
    right_bounds[i] = right_bounds[i + 1];
    right_bounds[i].grow(bins[i].bounds);
  }

  // Build left split bounds incrementally from left to right, keeping track of best split
  std::vector<AABB> left_bounds(num_bins);
  left_bounds.front() = bins.front().bounds;
  uint32_t left_num_triangles = bins.front().num_triangles;

  for (uint32_t split_index = 0; split_index < num_splits; split_index++) {
    uint32_t right_num_triangles = num_triangles - left_num_triangles;
    const AABB& left = left_bounds[split_index];

    // Check not useless split
    if (left_num_triangles > 2 && right_num_triangles > 2) {
      const AABB& right = right_bounds[split_index];

      float left_cost = left.get_cost(left_num_triangles);
      float right_cost = right.get_cost(right_num_triangles);
      float total_cost = left_cost + right_cost;

      best_params = best_params.min({
        total_cost,
        split_index,
        axis,
        left,
        right,
        left_num_triangles,
        right_num_triangles,
      });
    }

    // Build next left bound
    left_bounds[split_index + 1] = left;
    left_bounds[split_index + 1].grow(bins[split_index + 1].bounds);
    left_num_triangles += bins[split_index + 1].num_triangles;
  }

  return best_params;
}

std::pair<std::unique_ptr<BVHNode>, std::unique_ptr<BVHNode>>
BVH::split_node(std::unique_ptr<BVHNode>& node,
                SplitParams&& best_params,
                const std::vector<std::pair<AABB, glm::uvec3>>& bound_splits) {
  // Create real nodes and push triangles in each node
  auto left = std::make_unique<BVHNode>();
  auto right = std::make_unique<BVHNode>();
  left->aabb = std::move(best_params.left);
  right->aabb = std::move(best_params.right);
  left->triangles.reserve(best_params.left_num_triangles);
  right->triangles.reserve(best_params.right_num_triangles);

  // Push triangle in left or right depending on center
  for (size_t i = 0; i < node->triangles.size(); i++) {
    const auto& triangle = node->triangles[i];
    uint32_t split_index = bound_splits[i].second[best_params.axis];

    if (split_index <= best_params.split_index) {
      left->triangles.emplace_back(std::move(triangle));
    } else {
      right->triangles.emplace_back(std::move(triangle));
    }
  }
  // Clear triangles from parent
  node->triangles.clear();

  return { std::move(left), std::move(right) };
}

void BVH::build_bvh_node(std::unique_ptr<BVHNode>& node, const int depth) {
  if (node->triangles.size() <= MIN_TRIANGLES_PER_LEAF) {
    return;
  }

  SplitParams best_params = SplitParams::make_default();
  best_params.cost = node->get_cost();

  // Reduce number of bins according to depth
  const uint32_t num_bins = static_cast<uint32_t>(MAX_BINS / (depth + 1));
  if (num_bins <= 1) {
    return;
  }
  const uint32_t num_splits = num_bins - 1;
  const glm::vec3 bin_start = node->aabb.bottom;
  const glm::vec3 bin_end = node->aabb.top;
  const glm::vec3 bin_step = (bin_end - bin_start) / static_cast<float>(num_bins);
  const glm::vec3 inv_bin_step = 1.0f / bin_step;

  // Precompute triangle bounds and split indices
  std::vector<std::pair<AABB, glm::uvec3>> bound_splits;
  std::transform(node->triangles.cbegin(), node->triangles.cend(), std::back_inserter(bound_splits),
                 [&](const auto& tri) {
                   AABB bounds = tri.get_bounds();
                   glm::vec3 center = bounds.get_center();
                   glm::uvec3 split_index = glm::min(
                     static_cast<glm::uvec3>((center - bin_start) * inv_bin_step), num_splits);
                   return std::make_pair(bounds, split_index);
                 });

  // Find best axis to split
  for (int axis = 0; axis < 3; axis++) {
    // Don't want triangles to be concentrated on axis
    if (std::abs(bin_end[axis] - bin_start[axis]) < 1e-4f) {
      continue;
    }

    // Create bins of AABBs of triangles
    std::vector<Bin> bins(num_bins, { AABB::make_no_intersection(), 0 });

    // Loop through triangles and put them in bins
    for (const auto& [bounds, split_index] : bound_splits) {
      auto& [bin_bounds, num_triangles] = bins[split_index[axis]];
      bin_bounds.grow(bounds);
      num_triangles++;
    }

    // Find best object split in current axis
    best_params = best_params.min(find_object_split(node, axis, bins));
  }

  // If no better split, this node is a leaf node
  if (best_params.axis == -1) {
    return;
  }

  auto [left, right] = split_node(node, std::move(best_params), bound_splits);

// Recursively build left and right nodes
#pragma omp task default(none) shared(left)
  build_bvh_node(left, depth + 1);
#pragma omp task default(none) shared(right)
  build_bvh_node(right, depth + 1);
#pragma omp taskwait

  node->left = std::move(left);
  node->right = std::move(right);
}

std::vector<FlatBVHNode> BVH::build_flat_bvh(std::unique_ptr<BVHNode>& root) {
  std::vector<FlatBVHNode> nodes;
  build_flat_bvh_vec(nodes, root);
  return nodes;
}

size_t BVH::build_flat_bvh_vec(std::vector<FlatBVHNode>& flat_nodes,
                               std::unique_ptr<BVHNode>& node) {
  if (!node) {
    return 0;
  }

  size_t flat_node_index = flat_nodes.size();

  // Build flat node and insert into list
  FlatBVHNode flat_node { { node->aabb.top.x, node->aabb.top.y, node->aabb.top.z, 0 },
                          { node->aabb.bottom.x, node->aabb.bottom.y, node->aabb.bottom.z, 0 } };
  flat_nodes.emplace_back(std::move(flat_node));

  // Leaf node
  if (!node->triangles.empty()) {
    assert(!node->left && !node->right);

    // Denote that the node is a leaf node by negating
    w(flat_nodes[flat_node_index].top_offset_left) = triangles.size();
    w(flat_nodes[flat_node_index].bottom_num_right) = -static_cast<float>(node->triangles.size());
    triangles.insert(triangles.end(), std::make_move_iterator(node->triangles.begin()),
                     std::make_move_iterator(node->triangles.end()));
    node->triangles.clear();
  } else { // Inner node
    assert(node->left && node->right);

    // Recursively build left and right nodes and attach to parent
    w(flat_nodes[flat_node_index].top_offset_left) = build_flat_bvh_vec(flat_nodes, node->left);
    w(flat_nodes[flat_node_index].bottom_num_right) = build_flat_bvh_vec(flat_nodes, node->right);
  }

  return flat_node_index;
}

std::istream& operator>>(std::istream& in, FlatBVHNode& node) {
  in >> std::hex;
  float4* elems[] = { &node.top_offset_left, &node.bottom_num_right };

  for (int e = 0; e < 2; e++) {
    uint32_t s[4];
    float* f = reinterpret_cast<float*>(s);
    in >> s[0] >> s[1] >> s[2] >> s[3];
    x(*elems[e]) = f[0];
    y(*elems[e]) = f[1];
    z(*elems[e]) = f[2];
    w(*elems[e]) = f[3];
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const FlatBVHNode& node) {
  out << std::hex;
  const float4* elems[] = { &node.top_offset_left, &node.bottom_num_right };

  for (int e = 0; e < 2; e++) {
    float f[4];
    f[0] = x(*elems[e]);
    f[1] = y(*elems[e]);
    f[2] = z(*elems[e]);
    f[3] = w(*elems[e]);
    uint32_t* s = reinterpret_cast<uint32_t*>(&f);
    out << s[0] << " " << s[1] << " " << s[2] << " " << s[3] << " ";
  }
  return out;
}
