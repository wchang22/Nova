#ifndef CUDA_KERNEL_INTERSECTION_H
#define CUDA_KERNEL_INTERSECTION_H

#include "types.h"
#include "kernel_types/triangle.h"

__device__
bool intersects_triangle(const Ray& ray, Intersection& intrs, int tri_index,
                         const TriangleData& tri);
__device__
bool intersects_aabb(const Ray& ray, float3 top, float3 bottom);

#endif // CUDA_KERNEL_INTERSECTION_H