#include "accelerator.hpp"
#include "constants.hpp"

namespace nova {

Accelerator::Accelerator() {
  kernel_constants = { TRIANGLES_PER_LEAF_BITS, TRIANGLE_NUM_SHIFT, TRIANGLE_OFFSET_MASK };
}

}
