#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// TRIANGLES_PER_LEAF_BITS is passed in through opencl build options
#ifndef TRIANGLES_PER_LEAF_BITS
  #define TRIANGLES_PER_LEAF_BITS 6
#endif

constant uint STACK_SIZE = 96;

constant uint TRIANGLE_NUM_SHIFT = 32 - TRIANGLES_PER_LEAF_BITS;
constant uint TRIANGLE_OFFSET_MASK =
  (0xFFFFFFFF << TRIANGLES_PER_LEAF_BITS) >> TRIANGLES_PER_LEAF_BITS;

constant float3 LIGHT_POS = { -4, 2.8, 7 };

constant int SHININESS = 32;
constant float3 DEFAULT_AMBIENT = 0.1;
constant float3 DEFAULT_DIFFUSE = 0.4;
constant float3 DEFAULT_SPECULAR = 0.4;

constant float RAY_EPSILON = 1e-2; // Prevent self-shadowing

constant uint RAY_RECURSION_DEPTH = 5;

#endif // CONFIGURATION_H
