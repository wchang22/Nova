#ifndef BVH_H
#define BVH_H

#include <CL/cl2.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <limits>
#include <fstream>

#include "intersectables/triangle.h"

using namespace glm;

const vec3 VEC_MAX(std::numeric_limits<float>::max());

struct BVHNode {
  vec3 top;
  vec3 bottom;
  std::vector<Triangle> triangles;
  std::unique_ptr<BVHNode> left;
  std::unique_ptr<BVHNode> right;

  BVHNode() : top(-VEC_MAX), bottom(VEC_MAX) {};
};

// Packed to 32 bytes to fit in a cache line
// The 4th element of top_offset_left contains either the triangle offset or the left index
// The 4th element of bottom_num_right contains either the number of triangles or the right index
// depending on whether or not the node is an inner node or a leaf node
// If the 4th element of top_offset_left < 0, then the node is a leaf node
struct FlatBVHNode {
  cl_float4 top_offset_left;
  cl_float4 bottom_num_right;
};

std::istream& operator>>(std::istream& in, FlatBVHNode& node);
std::ostream& operator<<(std::ostream& out, const FlatBVHNode& node);

class BVH {
public:
  BVH(std::vector<Triangle>& triangles);

  // Note: Modifies `triangles`
  cl::Buffer build_bvh_buffer(const cl::Context& context);
private:
  std::unique_ptr<BVHNode> build_bvh();
  std::vector<FlatBVHNode> build_flat_bvh(std::unique_ptr<BVHNode>& root);
  void build_bvh_node(std::unique_ptr<BVHNode>& node, int depth);
  size_t build_flat_bvh_vec(std::vector<FlatBVHNode>& flat_nodes, std::unique_ptr<BVHNode>& node);

  std::vector<Triangle>& triangles;
};

#endif