#ifndef INTERSECTABLE_MANAGER_H
#define INTERSECTABLE_MANAGER_H

#include <vector>
#include <CL/cl2.hpp>

#include "intersectables/triangle.h"
#include "intersectables/material.h"

class IntersectableManager {
public:
  void add_triangle(const Triangle& tri, const Material& mat);
  std::pair<cl::Buffer, size_t> build_buffer(const cl::Context& context);

private:
  std::vector<std::pair<TriangleData, MaterialData>> triangles;
};

#endif // INTERSECTABLE_MANAGER_H