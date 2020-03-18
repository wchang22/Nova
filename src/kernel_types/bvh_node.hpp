#ifndef KERNEL_TYPE_BVH_NODE_HPP
#define KERNEL_TYPE_BVH_NODE_HPP

#include "backend/types.hpp"

// Packed to 32 bytes to fit in a cache line
// The 4th element of top_offset_left contains either the triangle offset or the left index
// The 4th element of bottom_num_right contains either the number of triangles or the right index
// depending on whether or not the node is an inner node or a leaf node
// If the 4th element of bottom_num_right < 0, then the node is a leaf node
struct FlatBVHNode {
  float4 top_offset_left;
  float4 bottom_num_right;
};

#endif // KERNEL_TYPE_BVH_NODE_HPP
