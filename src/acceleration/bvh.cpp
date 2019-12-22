#include <glm/gtx/vec_swizzle.hpp>
#include <cassert>

#include "bvh.h"

// Algorithm from https://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

constexpr size_t MIN_TRIANGLES_PER_LEAF = 15;
constexpr float MAX_BINS = 1024.f;

BVH::BVH(std::vector<Triangle>& triangles)
  : triangles(triangles)
{
}

cl::Buffer BVH::build_bvh_buffer(const cl::Context& context) {
  std::unique_ptr<BVHNode> bvh = build_bvh();
  std::vector<FlatBVHNode> flat_bvh = build_flat_bvh(bvh);

  cl::Buffer buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                 flat_bvh.size() * sizeof(decltype(flat_bvh)::value_type),
                 flat_bvh.data());
  return buf;
}

std::unique_ptr<BVHNode> BVH::build_bvh() {
  auto root = std::make_unique<BVHNode>();

  root->triangles.reserve(triangles.size());

  // Create bounding box to capture all triangles
  for (const auto& triangle : triangles) {
    root->triangles.emplace_back(std::move(triangle));

    auto [top, bottom] = triangle.get_bounds();
    root->top = max(top, root->top);
    root->bottom = min(bottom, root->bottom);
  }
  // Clear triangles, as they have all been moved out
  triangles.clear();

  // Recursively build bvh
  build_bvh_node(root, 0);
  return root;
}

float node_cost(BVHNode& node, size_t num_triangles) {
  vec3 dims = abs(node.top - node.bottom);
  float surface_area = dot(xyz(dims), yzx(dims));

  return num_triangles * surface_area;
}

void BVH::build_bvh_node(std::unique_ptr<BVHNode>& node, int depth) {
  if (node->triangles.size() <= MIN_TRIANGLES_PER_LEAF) {
    return;
  }

  float min_cost = node_cost(*node, node->triangles.size());
  float best_split = std::numeric_limits<float>::max();
  int best_axis = -1;

  // Find best axis to split
  for (int axis = 0; axis < 3; axis++) {
    float bin_start = node->bottom[axis];
    float bin_end = node->top[axis];

    // Don't want triangles to be concentrated on axis
    if (abs(bin_end - bin_start) < 1e-4f) {
      continue;
    }

    // Reduce number of bins according to depth
    float bin_step = (bin_end - bin_start) / (MAX_BINS / (depth + 1));

    // Find best split (split with least total cost)
    for (float split = bin_start + bin_step; split < bin_end - bin_step; split += bin_step) {
      BVHNode left, right;

      uint32_t left_num_triangles = 0;
      uint32_t right_num_triangles = 0;

      // Try putting each triangle in either the left or right node based on center
      for (const auto& triangle : node->triangles) {
        auto [top, bottom] = triangle.get_bounds();
        float center = ((top + bottom) / 2.f)[axis];

        if (center < split) {
          left.top = max(top, left.top);
          left.bottom = min(bottom, left.bottom);
          left_num_triangles++;
        } else {
          right.top = max(top, right.top);
          right.bottom = min(bottom, right.bottom);
          right_num_triangles++;
        }
      }

      // Useless splits
      if (left_num_triangles <= 1 || right_num_triangles <= 1) {
        continue;
      }

      float left_cost = node_cost(left, left_num_triangles);
      float right_cost = node_cost(right, right_num_triangles);
      float total_cost = left_cost + right_cost;

      if (total_cost < min_cost) {
        min_cost = total_cost;
        best_split = split;
        best_axis = axis;
      }
    }
  }

  // If no better split, this node is a leaf node
  if (best_axis == -1) {
    return;
  }

  // Create real nodes and push triangles in each node
  auto left = std::make_unique<BVHNode>();
  auto right = std::make_unique<BVHNode>();

  for (const auto& triangle : node->triangles) {
    auto [top, bottom] = triangle.get_bounds();
    float center = ((top + bottom) / 2.f)[best_axis];

    if (center < best_split) {
      left->top = max(top, left->top);
      left->bottom = min(bottom, left->bottom);
      left->triangles.emplace_back(std::move(triangle));
    } else {
      right->top = max(top, right->top);
      right->bottom = min(bottom, right->bottom);
      right->triangles.emplace_back(std::move(triangle));
    }
  }
  // Clear triangles from parent
  node->triangles.clear();

  // Recursively build left and right nodes
  build_bvh_node(left, depth + 1);
  build_bvh_node(right, depth + 1);

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
  FlatBVHNode flat_node {
    { {node->top.x, node->top.y, node->top.z, 0} },
    { {node->bottom.x, node->bottom.y, node->bottom.z, 0} }
  };
  flat_nodes.emplace_back(std::move(flat_node));

  // Leaf node
  if (!node->triangles.empty()) {
    assert(!node->left && !node->right);

    // Denote that the node is a leaf node by negating
    flat_nodes[flat_node_index].top_offset_left.s[3] = -static_cast<float>(triangles.size());
    flat_nodes[flat_node_index].bottom_num_right.s[3] = node->triangles.size();
    // TODO: Sort?
    triangles.insert(triangles.end(),
                     std::make_move_iterator(node->triangles.begin()),
                     std::make_move_iterator(node->triangles.end()));
    node->triangles.clear();
  } else { // Inner node
    assert(node->left || node->right);

    // Recursively build left and right nodes and attach to parent
    flat_nodes[flat_node_index].top_offset_left.s[3] = build_flat_bvh_vec(flat_nodes, node->left);
    flat_nodes[flat_node_index].bottom_num_right.s[3] = build_flat_bvh_vec(flat_nodes, node->right);
  }

  return flat_node_index;
}
