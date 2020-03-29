#ifndef KERNEL_TYPE_KERNEL_CONSTANTS_HPP
#define KERNEL_TYPE_KERNEL_CONSTANTS_HPP

#include "backend/types.hpp"

namespace nova {

struct KernelConstants {
  int triangle_per_leaf_bits;
  unsigned triangle_num_shift;
  unsigned triangle_offset_mask;
};

}

#endif // KERNEL_TYPE_KERNEL_CONSTANTS_HPP