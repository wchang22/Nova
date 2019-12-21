#ifndef BVH_H
#define BVH_H

#include <CL/cl2.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <limits>

#include "intersectables/triangle.h"

using namespace glm;

constexpr vec3 VEC_MAX(std::numeric_limits<float>::max());

struct BVHNode {
  vec3 top;
  vec3 bottom;
  std::vector<Triangle> triangles;
  std::unique_ptr<BVHNode> left;
  std::unique_ptr<BVHNode> right;

  BVHNode() : top(-VEC_MAX), bottom(VEC_MAX) {};
};

// TODO: Should be packed to 32 bytes
struct FlatBVHNode {
  cl_float3 top;
  cl_float3 bottom;
  cl_uint triangle_offset;
  cl_uint num_triangles;
  cl_int left;
  cl_int right;
};

class BVH {
public:
  BVH(std::vector<Triangle>& triangles);

  // Note: Modifies `triangles`
  cl::Buffer build_bvh_buffer(const cl::Context& context);
private:
  std::unique_ptr<BVHNode> build_bvh();
  std::vector<FlatBVHNode> build_flat_bvh(std::unique_ptr<BVHNode>& root);
  void build_bvh_node(std::unique_ptr<BVHNode>& node, int depth);
  int build_flat_bvh_vec(std::vector<FlatBVHNode>& flat_nodes, std::unique_ptr<BVHNode>& node);

  std::vector<Triangle>& triangles;
};

#endif