#ifndef INTERSECTABLE_MANAGER_H
#define INTERSECTABLE_MANAGER_H

#include <vector>
#include <unordered_map>
#include <CL/cl2.hpp>

#include "intersectables/triangle.h"
#include "intersectables/material.h"

class IntersectableManager {
public:
  void add_triangle(const Triangle& tri, const Material& mat);
  void build_buffers(const cl::Context& context,
                     cl::Buffer& triangle_buf,
                     cl::Buffer& tri_normal_buf,
                     cl::Buffer& materials_buf,
                     cl::Buffer& bvh_buf);

private:
  std::vector<Triangle> triangles;
  std::unordered_map<Triangle, Material, TriangleHash> triangle_map;
};

#endif // INTERSECTABLE_MANAGER_H
