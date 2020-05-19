#ifndef CONSTANTS_CL
#define CONSTANTS_CL

// Data passed in through opencl build options
#ifndef TRIANGLES_PER_LEAF_BITS
  #define TRIANGLES_PER_LEAF_BITS 6
#endif
#ifndef TRIANGLE_NUM_SHIFT
  #define TRIANGLE_NUM_SHIFT 26
#endif
#ifndef TRIANGLE_OFFSET_MASK
  #define TRIANGLE_OFFSET_MASK 0x3FFFFFF
#endif

#define STACK_SIZE 96
#define SQRT3_3 0.577350269f

constant float RAY_EPSILON = 1e-2f; // Prevent self-shadowing
// Min epsilon to produce significant change in 8 bit colour channels
constant float COLOR_EPSILON = 0.5f / 255.0f;
// Min neighbour colour difference required to raytrace instead of interpolate
constant float INTERP_THRESHOLD = SQRT3_3;

// Anti-aliasing edge thresholds
constant float EDGE_THRESHOLD_MIN = 0.0312f;
constant float EDGE_THRESHOLD_MAX = 0.125f;
constant uint EDGE_SEARCH_ITERATIONS = 12;
constant float SUBPIXEL_QUALITY = 0.75f;

constant sampler_t image_sampler =
  CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

constant sampler_t image_sampler_norm =
  CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;

#endif // CONSTANTS_CL
