#ifndef INTERSECTABLE_MANAGER_H
#define INTERSECTABLE_MANAGER_H

#include <vector>
#include <CL/cl2.hpp>

#include "intersectables/triangle.h"

class IntersectableManager {
public:
  void add_triangle(const Triangle& tri);
  std::pair<cl::Buffer, size_t> build_buffer(const cl::Context& context);

private:
  std::vector<TriangleData> triangles;
};

#endif // INTERSECTABLE_MANAGER_H