#include <glm/gtx/vec_swizzle.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <algorithm>

#include "bvh.h"
#include "util/exception/exception.h"
#include "util/serialization/serialization.h"
#include "constants.h"

enum class SplitType : uint8_t {
  LEFT,
  RIGHT,
  BOTH,
};

struct SplitParams {
  float cost;
  float split;
  int axis;
  AABB left;
  AABB right;
  size_t left_num_triangles;
  size_t right_num_triangles;
  std::vector<SplitType> split_type;

  static SplitParams make_default() {
    return { std::numeric_limits<float>::max(), 0, -1, {}, {}, 0, 0, {} };
  }

  SplitParams min(SplitParams&& other) {
    return cost < other.cost ? std::move(*this) : std::move(other);
  }
};

const AABB NO_INTERSECTION { -VEC_MAX, VEC_MAX };

struct Bin {
  AABB bound;
  int enter;
  int exit;
};

// Algorithm from https://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

BVH::BVH(const std::string& name, std::vector<Triangle>& triangles)
  : name(name), triangles(triangles)
{
}

std::vector<FlatBVHNode> BVH::build() {
  std::string bvh_file_name = name + ".bvh";
  std::string tri_file_name = name + ".tri";
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

  // Recursively build bvh
  build_bvh_node(root, 0, root->aabb.get_surface_area());
  return root;
}

SplitParams get_object_split(int axis, float split, std::unique_ptr<BVHNode>& node) {
  AABB left { -VEC_MAX, VEC_MAX };
  AABB right { -VEC_MAX, VEC_MAX };

  uint32_t left_num_triangles = 0;
  uint32_t right_num_triangles = 0;

  std::vector<SplitType> split_type(node->triangles.size());

  // Try putting each triangle in either the left or right node based on center
  for (size_t i = 0; i < node->triangles.size(); i++) {
    const auto& triangle = node->triangles[i];
    AABB bounds = triangle.get_bounds();
    float center = bounds.get_center()[axis];

    if (center < split) {
      left.grow(bounds);
      left_num_triangles++;
      split_type[i] = SplitType::LEFT;
    } else {
      right.grow(bounds);
      right_num_triangles++;
      split_type[i] = SplitType::RIGHT;
    }
  }

  // Useless splits
  if (left_num_triangles <= 1 || right_num_triangles <= 1) {
    return SplitParams::make_default();
  }

  float left_cost = left.get_cost(left_num_triangles);
  float right_cost = right.get_cost(right_num_triangles);
  float total_cost = left_cost + right_cost;

  return {
    total_cost,
    split,
    axis,
    left,
    right,
    left_num_triangles,
    right_num_triangles,
    split_type,
  };
}

SplitParams get_spatial_split(int axis, float split, size_t index,
                              std::unique_ptr<BVHNode>& node,
                              const std::vector<Bin>& bins,
                              const std::vector<AABB>& clips) {
  AABB left = NO_INTERSECTION;
  AABB right = NO_INTERSECTION;

  uint32_t left_num_triangles = 0;
  uint32_t right_num_triangles = 0;

  std::vector<SplitType> split_type(node->triangles.size());

  for (size_t i = 0; i <= index; i++) {
    const auto& [bounds, enter, exit] = bins[i];
    left_num_triangles += enter;
    left.grow(clips[i]);
  }
  for (size_t i = index + 1; i < bins.size(); i++) {
    const auto& [bounds, enter, exit] = bins[i];
    right_num_triangles += exit;
    right.grow(clips[i]);
  }

  // AABB left_clip = node->aabb;
  // AABB right_clip = node->aabb;
  // left_clip.top[axis] = split;
  // right_clip.bottom[axis] = split;
  // left = left_clip;
  // right = right_clip;

  // Try putting each triangle in either the left or right node based on center
  for (size_t i = 0; i < node->triangles.size(); i++) {
    const auto& triangle = node->triangles[i];
    AABB bounds = triangle.get_bounds();

    if (bounds.top[axis] <= split) {
      assert(bounds.intersects(left));
      split_type[i] = SplitType::LEFT;
    } else if (bounds.bottom[axis] >= split) {
      assert(bounds.intersects(right));
      split_type[i] = SplitType::RIGHT;
    } else {
      assert((bounds.intersects(left)) &&
             (bounds.intersects(right)));
      split_type[i] = SplitType::BOTH;
    }
  }

  // Useless splits
  if (left_num_triangles <= 1 || right_num_triangles <= 1) {
    return SplitParams::make_default();
  }

  float left_cost = left.get_cost(left_num_triangles);
  float right_cost = right.get_cost(right_num_triangles);
  float total_cost = left_cost + right_cost;

  return {
    total_cost,
    split,
    axis,
    left,
    right,
    left_num_triangles,
    right_num_triangles,
    split_type,
  };
}

void BVH::build_bvh_node(std::unique_ptr<BVHNode>& node, int depth, const float root_sa) {
  if (node->triangles.size() <= MIN_TRIANGLES_PER_LEAF) {
    return;
  }

  SplitParams best_params = SplitParams::make_default();
  best_params.cost = node->get_cost();

  // Find best axis to split
  for (int axis = 0; axis < 3; axis++) {
    const float bin_start = node->aabb.bottom[axis];
    const float bin_end = node->aabb.top[axis];

    // Don't want triangles to be concentrated on axis
    if (abs(bin_end - bin_start) < 1e-4f) {
      continue;
    }

    // Reduce number of bins according to depth
    const size_t num_bins = MAX_BINS / (depth + 1);
    const size_t num_splits = num_bins - 1;
    const float bin_step = (bin_end - bin_start) / num_bins;
    const float inv_bin_step = 1.0f / bin_step;

    // Find best split (split with least total cost)
    #pragma omp declare reduction \
      (param_min:SplitParams:omp_out=omp_out.min(std::move(omp_in))) \
      initializer(omp_priv=SplitParams::make_default())

    #pragma omp parallel for default(none) \
      shared(node, axis) \
      reduction(param_min:best_params) \
      schedule(dynamic)
    for (size_t i = 0; i < num_splits; i++) {
      float split = bin_start + (i + 1) * bin_step;
      best_params = best_params.min(get_object_split(axis, split, node));
    }

    // AABBs overlap
    if (best_params.left.intersects(best_params.right)) {
      AABB intersection = best_params.left.get_intersection(best_params.right);
      float intersection_sa = intersection.get_surface_area();

      // Only compute spatial split if overlap is significant
      if (intersection_sa / root_sa > OVERLAP_TOLERANCE) {
        std::vector<Bin> bins(num_bins, { NO_INTERSECTION, 0, 0 });
        std::vector<AABB> clips(num_bins);

        for (size_t i = 0; i < num_bins; i++) {
          AABB clip = node->aabb;         
          clip.bottom[axis] = bin_start + i * bin_step;
          clip.top[axis] = bin_start + (i + 1) * bin_step;
          clips[i] = std::move(clip);
        }

        for (const auto& triangle : node->triangles) {
          AABB bounds = triangle.get_bounds();
          assert(bounds.top[axis] >= bounds.bottom[axis]);
          size_t start =
            std::clamp((bounds.bottom[axis] - bin_start) * inv_bin_step,
                        0.0f, static_cast<float>(num_bins - 1));
          size_t end =
            std::clamp((bounds.top[axis] - bin_start) * inv_bin_step,
                        static_cast<float>(start), static_cast<float>(num_bins - 1));
          assert(bounds.intersects(node->aabb));

          for (size_t i = start; i <= end; i++) {
            bins[i].bound.grow(triangle.get_clipped_bounds(clips[i]));
          }

          bins[start].enter++;
          bins[end].exit++;
        }

        for (size_t i = 0; i < num_splits; i++) {
          float split = bin_start + (i + 1) * bin_step;
          best_params = best_params.min(get_spatial_split(axis, split, i, node, bins, clips));
        }
      }
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

  for (size_t i = 0; i < node->triangles.size(); i++) {
    const auto& triangle = node->triangles[i];
    AABB bounds = triangle.get_bounds();
    switch (best_params.split_type[i]) {
      case SplitType::LEFT:
        assert(bounds.intersects(left->aabb));
        left->triangles.emplace_back(std::move(triangle));
        break;
      case SplitType::RIGHT:
        assert(bounds.intersects(right->aabb));
        right->triangles.emplace_back(std::move(triangle));
        break;
      case SplitType::BOTH:
        assert(bounds.intersects(left->aabb) && bounds.intersects(right->aabb));
        left->triangles.emplace_back(triangle);
        right->triangles.emplace_back(std::move(triangle));
        break;
    }
  }
  // Clear triangles from parent
  node->triangles.clear();

  // Recursively build left and right nodes
  build_bvh_node(left, depth + 1, root_sa);
  build_bvh_node(right, depth + 1, root_sa);

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
    { node->aabb.top.x, node->aabb.top.y, node->aabb.top.z, 0 },
    { node->aabb.bottom.x, node->aabb.bottom.y, node->aabb.bottom.z, 0 }
  };
  flat_nodes.emplace_back(std::move(flat_node));

  // Leaf node
  if (!node->triangles.empty()) {
    assert(!node->left && !node->right);

    // Denote that the node is a leaf node by negating
    w(flat_nodes[flat_node_index].top_offset_left) = triangles.size();
    w(flat_nodes[flat_node_index].bottom_num_right) = -static_cast<float>(node->triangles.size());
    triangles.insert(triangles.end(),
                     std::make_move_iterator(node->triangles.begin()),
                     std::make_move_iterator(node->triangles.end()));
    node->triangles.clear();
  } else { // Inner node
    assert(node->left || node->right);

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
