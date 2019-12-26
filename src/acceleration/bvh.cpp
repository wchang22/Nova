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
    if (line.empty()) {
      throw FileException("Invalid bvh file " + bvh_file_name);
    }
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
    root->aabb.grow(triangle.get_bounds());
    root->triangles.emplace_back(std::move(triangle));
  }
  // Clear triangles, as they have all been moved out
  triangles.clear();

  // Recursively build bvh
  build_bvh_node(root, 0);
  return root;
}

void BVH::build_bvh_node(std::unique_ptr<BVHNode>& node, int depth) {
  if (node->triangles.size() <= MIN_TRIANGLES_PER_LEAF) {
    return;
  }

  SplitParams best_params {
    std::numeric_limits<float>::max(), 0, -1, {}, {}, 0, 0
  };

  // Find best axis to split
  for (int axis = 0; axis < 3; axis++) {
    float bin_start = node->aabb.bottom[axis];
    float bin_end = node->aabb.top[axis];

    // Don't want triangles to be concentrated on axis
    if (abs(bin_end - bin_start) < 1e-4f) {
      continue;
    }

    // Reduce number of bins according to depth
    float bin_step = (bin_end - bin_start) / (MAX_BINS / (depth + 1));
    int num_bins = MAX_BINS / (depth + 1) - 2;

    // Find best split (split with least total cost)
    #pragma omp declare reduction \
      (param_min:SplitParams:omp_out=omp_out.min(omp_in)) \
      initializer(omp_priv={ \
        std::numeric_limits<float>::max(), 0, -1, {}, {}, 0, 0 \
      })

    #pragma omp parallel for default(none) \
      shared(node, axis, bin_start, bin_end, bin_step, num_bins, VEC_MAX) \
      reduction(param_min:best_params) \
      schedule(dynamic)
    for (int i = 0; i < num_bins; i++) {
      float split = bin_start + (i + 1) * bin_step;

      AABB left { -VEC_MAX, VEC_MAX };
      AABB right { -VEC_MAX, VEC_MAX };

      uint32_t left_num_triangles = 0;
      uint32_t right_num_triangles = 0;

      // Try putting each triangle in either the left or right node based on center
      for (const auto& triangle : node->triangles) {
        AABB bounds = triangle.get_bounds();
        float center = bounds.get_center()[axis];

        if (center < split) {
          left.grow(bounds);
          left_num_triangles++;
        } else {
          right.grow(bounds);
          right_num_triangles++;
        }
      }

      // Useless splits
      if (left_num_triangles <= 1 || right_num_triangles <= 1) {
        continue;
      }

      float left_cost = left.get_cost(left_num_triangles);
      float right_cost = right.get_cost(right_num_triangles);
      float total_cost = left_cost + right_cost;

      SplitParams params {
        total_cost,
        split,
        axis,
        left,
        right,
        left_num_triangles,
        right_num_triangles,
      };
      best_params = best_params.min(params);
    }
  }

  // If no better split, this node is a leaf node
  if (best_params.axis == -1) {
    return;
  }

  // Create real nodes and push triangles in each node
  auto left = std::make_unique<BVHNode>();
  auto right = std::make_unique<BVHNode>();
  left->aabb = best_params.left;
  right->aabb = best_params.right;
  left->triangles.reserve(best_params.left_num_triangles);
  right->triangles.reserve(best_params.right_num_triangles);

  for (const auto& triangle : node->triangles) {
    float center = triangle.get_bounds().get_center()[best_params.axis];

    if (center < best_params.split) {
      left->triangles.emplace_back(std::move(triangle));
    } else {
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
    { {node->aabb.top.x, node->aabb.top.y, node->aabb.top.z, 0} },
    { {node->aabb.bottom.x, node->aabb.bottom.y, node->aabb.bottom.z, 0} }
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
