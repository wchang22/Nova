#ifndef KERNEL_RANDOM_HPP
#define KERNEL_RANDOM_HPP

#include "kernels/backend/kernel.hpp"

namespace nova {

// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
DEVICE inline uint wang_hash(uint seed) {
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}

DEVICE inline float xorshift_rand(uint& rng_state) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state / static_cast<float>(UINT_MAX);
}

}

#endif // KERNEL_RANDOM_HPP
