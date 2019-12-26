#include <glm/gtx/vec_swizzle.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cassert>
#include <filesystem>

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

SplitParams get_spatial_split(int axis, float split, std::unique_ptr<BVHNode>& node) {
  AABB left = NO_INTERSECTION;
  AABB right = NO_INTERSECTION;

  AABB left_clip = node->aabb;
  AABB right_clip = node->aabb;
  left_clip.top[axis] = split;
  right_clip.bottom[axis] = split;

  uint32_t left_num_triangles = 0;
  uint32_t right_num_triangles = 0;

  std::vector<SplitType> split_type(node->triangles.size());

  // Try putting each triangle in either the left or right node based on center
  for (size_t i = 0; i < node->triangles.size(); i++) {
    const auto& triangle = node->triangles[i];
    AABB bounds = triangle.get_bounds();
    AABB left_bounds = triangle.get_clipped_bounds(left_clip);
    AABB right_bounds = triangle.get_clipped_bounds(right_clip);

    assert(all(lessThanEqual(left_bounds.top, left_clip.top)));
    assert(all(greaterThanEqual(left_bounds.bottom, left_clip.bottom)));
    assert(all(lessThanEqual(right_bounds.top, right_clip.top)));
    assert(all(greaterThanEqual(right_bounds.bottom, right_clip.bottom)));

    assert(all(greaterThanEqual(left.top, left.bottom)) || left == NO_INTERSECTION);
    assert(all(greaterThanEqual(right.top, right.bottom)) || right == NO_INTERSECTION);
    assert(bounds.intersects(node->aabb) || node->aabb.intersects(bounds));
    assert(!(left_bounds == NO_INTERSECTION) || !(right_bounds == NO_INTERSECTION));
    assert((bounds.intersects(left_clip) || left_clip.intersects(bounds)) ||
               (bounds.intersects(right_clip) || right_clip.intersects(bounds)));

    if (right_bounds == NO_INTERSECTION) {
      left.grow(left_bounds);
      assert((bounds.intersects(left) || left.intersects(bounds)));
      left_num_triangles++;
      split_type[i] = SplitType::LEFT;
    } else if (left_bounds == NO_INTERSECTION) {
      right.grow(right_bounds);
      assert((bounds.intersects(right) || right.intersects(bounds)));
      right_num_triangles++;
      split_type[i] = SplitType::RIGHT;
    } else {
      left.grow(left_bounds);
      right.grow(right_bounds);
      assert((bounds.intersects(left) || left.intersects(bounds)) &&
               (bounds.intersects(right) || right.intersects(bounds)));
      left_num_triangles++;
      right_num_triangles++;

      // Reference unsplit
      float left_cost = left.get_cost(left_num_triangles);
      float right_cost = right.get_cost(right_num_triangles);
      float split_cost = left_cost + right_cost;

      left_cost = left.get_union(bounds).get_cost(left_num_triangles);
      right_cost = right.get_cost(right_num_triangles - 1);
      float unsplit_left_cost = left_cost + right_cost;

      left_cost = left.get_cost(left_num_triangles - 1);
      right_cost = left.get_union(bounds).get_cost(right_num_triangles);
      float unsplit_right_cost = left_cost + right_cost;

      float min_cost = std::min(split_cost, std::min(unsplit_left_cost, unsplit_right_cost));

      if (min_cost == unsplit_left_cost) {
        left.grow(bounds);
        right_num_triangles--;
        split_type[i] = SplitType::LEFT;
      } else if (min_cost == unsplit_right_cost) {
        right.grow(bounds);
        left_num_triangles--;
        split_type[i] = SplitType::RIGHT;
      } else {
        split_type[i] = SplitType::BOTH;
      }
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
    total_cost * 1.02f,
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
    float bin_start = node->aabb.bottom[axis];
    float bin_end = node->aabb.top[axis];

    // Don't want triangles to be concentrated on axis
    if (abs(bin_end - bin_start) < 1e-4f) {
      continue;
    }

    // Reduce number of bins according to depth
    float bin_step = (bin_end - bin_start) / (MAX_BINS / (depth + 1));
    int num_bins = MAX_BINS / (depth + 1) - 1;

    // Find best split (split with least total cost)
    #pragma omp declare reduction \
      (param_min:SplitParams:omp_out=omp_out.min(std::move(omp_in))) \
      initializer(omp_priv=SplitParams::make_default())

    #pragma omp parallel for default(none) \
      shared(node, axis, bin_start, bin_end, bin_step, num_bins, root_sa) \
      reduction(param_min:best_params) \
      schedule(dynamic)
    for (int i = 0; i < num_bins; i++) {
      float split = bin_start + (i + 1) * bin_step;

      SplitParams object_split_params = get_object_split(axis, split, node);
      AABB left_aabb = object_split_params.left;
      AABB right_aabb = object_split_params.right;

      best_params = best_params.min(std::move(object_split_params));

      // AABBs overlap
      if (left_aabb.intersects(right_aabb)) {
        AABB intersection = left_aabb.get_intersection(right_aabb);
        float intersection_sa = intersection.get_surface_area();

        // Only compute spatial split if overlap is significant
        if (intersection_sa / root_sa > OVERLAP_TOLERANCE) {
          SplitParams spatial_split_params = get_spatial_split(axis, split, node);
          best_params = best_params.min(std::move(spatial_split_params));
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
    switch (best_params.split_type[i]) {
      case SplitType::LEFT:
        left->triangles.emplace_back(std::move(triangle));
        break;
      case SplitType::RIGHT:
        right->triangles.emplace_back(std::move(triangle));
        break;
      case SplitType::BOTH:
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
    assert(node->triangles.size() <= MAX_TRIANGLES_PER_LEAF);

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
