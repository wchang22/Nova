#include <glm/gtx/vec_swizzle.hpp>
#include <cassert>
#include <filesystem>

#include "bvh.h"
#include "util/exception/exception.h"
#include "util/serialization/serialization.h"
#include "configuration.h"

// Algorithm from https://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

BVH::BVH(std::vector<Triangle>& triangles)
  : triangles(triangles)
{
}

cl::Buffer BVH::build_bvh_buffer(const cl::Context& context) {
  std::string bvh_file_name = std::filesystem::path(MODEL_PATH).stem().string() + ".bvh";
  std::string tri_file_name = std::filesystem::path(MODEL_PATH).stem().string() + ".tri";
  std::fstream bvh_file(bvh_file_name);
  std::fstream tri_file(tri_file_name);

  std::vector<FlatBVHNode> flat_bvh;

  // If cached files do not exist
  if (!bvh_file.is_open() || !tri_file.is_open()) {
    bvh_file.open(bvh_file_name, std::ios::out);
    tri_file.open(tri_file_name, std::ios::out);
    if (!bvh_file.is_open()) {
      throw FileException("Cannot create " + bvh_file_name);
    }
    if (!tri_file.is_open()) {
      throw FileException("Cannot create " + tri_file_name);
    }

    // Build bvh and serialize them into files
    std::unique_ptr<BVHNode> bvh = build_bvh();
    flat_bvh = build_flat_bvh(bvh);

    bvh_file << flat_bvh;
    tri_file << triangles;
  } else {
    size_t triangles_size = triangles.size();

    // Each line is a bvh node
    std::string line; 
    getline(bvh_file, line);
    bvh_file.seekg(0);
    size_t bvh_size = std::filesystem::file_size(bvh_file_name) / line.length();

    triangles.clear();
    triangles.reserve(triangles_size);
    flat_bvh.reserve(bvh_size);

    // Deserialize bvh and triangles from file
    bvh_file >> flat_bvh;
    tri_file >> triangles;
  }

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

std::istream& operator>>(std::istream& in, FlatBVHNode& node) {
  in >> std::hex;
  cl_float4* elems[] = { &node.top_offset_left, &node.bottom_num_right };

  for (int e = 0; e < 2; e++) {
    for (int i = 0; i < 4; i++) {
      uint32_t x;
      float* f = reinterpret_cast<float*>(&x);
      in >> x;
      elems[e]->s[i] = *f;
    }
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const FlatBVHNode& node) {
  out << std::hex;
  const cl_float4* elems[] = { &node.top_offset_left, &node.bottom_num_right };

  for (int e = 0; e < 2; e++) {
    for (int i = 0; i < 4; i++) {
      float f = elems[e]->s[i];
      uint32_t* x = reinterpret_cast<uint32_t*>(&f);
      out << *x << " ";
    }
  }
  return out;
}
