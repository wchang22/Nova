#ifndef SHARED_CONSTANTS_HPP
#define SHARED_CONSTANTS_HPP

// BVH/Triangle constants
constexpr size_t TRIANGLES_PER_LEAF_BITS = 6;
constexpr size_t TRIANGLE_NUM_SHIFT = 32 - TRIANGLES_PER_LEAF_BITS;
constexpr size_t TRIANGLE_OFFSET_MASK =
  (0xFFFFFFFF << TRIANGLES_PER_LEAF_BITS) >> TRIANGLES_PER_LEAF_BITS;
constexpr size_t MIN_TRIANGLES_PER_LEAF = 8;
constexpr float MAX_BINS = 1024.f;
constexpr size_t MAX_TRIANGLES = (1 << (32 - TRIANGLES_PER_LEAF_BITS)) - 1;

static_assert(TRIANGLES_PER_LEAF_BITS <= 32);
static_assert(MIN_TRIANGLES_PER_LEAF < (1 << TRIANGLES_PER_LEAF_BITS));

#endif