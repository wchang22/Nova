#ifndef CONSTANTS_CL
#define CONSTANTS_CL

// Data passed in through opencl build options
#ifndef TRIANGLES_PER_LEAF_BITS
  #define TRIANGLES_PER_LEAF_BITS 6
#endif
#ifndef LIGHT_POS
  #define LIGHT_POS (float3)(-4, 2.8, 7)
#endif
#ifndef SHININESS
  #define SHININESS 32
#endif
#ifndef DEFAULT_AMBIENT
  #define DEFAULT_AMBIENT 0.1
#endif
#ifndef DEFAULT_DIFFUSE
  #define DEFAULT_DIFFUSE 0.4
#endif
#ifndef DEFAULT_SPECULAR
  #define DEFAULT_SPECULAR 0.4
#endif
#ifndef RAY_RECURSION_DEPTH
  #define RAY_RECURSION_DEPTH 5
#endif

constant uint STACK_SIZE = 96;

constant uint TRIANGLE_NUM_SHIFT = 32 - TRIANGLES_PER_LEAF_BITS;
constant uint TRIANGLE_OFFSET_MASK =
  (0xFFFFFFFF << TRIANGLES_PER_LEAF_BITS) >> TRIANGLES_PER_LEAF_BITS;

constant float RAY_EPSILON = 1e-2; // Prevent self-shadowing

#endif // CONSTANTS_CL
