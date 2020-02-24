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
#ifndef LIGHT_POSITION
  #define LIGHT_POSITION (float3)(-4.0, 2.8, 7.0)
#endif
#ifndef LIGHT_INTENSITY
  #define LIGHT_INTENSITY (float3)(500.0)
#endif
#ifndef DEFAULT_DIFFUSE
  #define DEFAULT_DIFFUSE (float3)(1.0, 1.0, 1.0)
#endif
#ifndef DEFAULT_METALLIC
  #define DEFAULT_METALLIC 0.5
#endif
#ifndef DEFAULT_ROUGHNESS
  #define DEFAULT_ROUGHNESS 0.1
#endif
#ifndef DEFAULT_AMBIENT_OCCLUSION
  #define DEFAULT_AMBIENT_OCCLUSION 0.1
#endif
#ifndef RAY_RECURSION_DEPTH
  #define RAY_RECURSION_DEPTH 5
#endif

#define STACK_SIZE 96

constant float RAY_EPSILON = 1e-2f; // Prevent self-shadowing
// Min epsilon to produce significant change in 8 bit colour channels
constant float COLOR_EPSILON = 0.5f / 255.0f; 

#endif // CONSTANTS_CL
