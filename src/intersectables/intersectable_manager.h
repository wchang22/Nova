#ifndef INTERSECTABLE_MANAGER_H
#define INTERSECTABLE_MANAGER_H

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

#include <vector>
#include <unordered_map>

#include "model/model.h"
#include "intersectables/triangle.h"

class IntersectableManager {
public:
  IntersectableManager(const std::string& name);
  void add_triangle(const Triangle& tri, const TriangleMeta& meta);
  void add_model(const Model& model);
  void build_buffers(const cl::Context& context,
                     cl::Buffer& triangle_buf,
                     cl::Buffer& tri_meta_buf,
                     cl::Buffer& bvh_buf);

private:
  std::string name;
  std::vector<Triangle> triangles;
  std::unordered_map<Triangle, TriangleMeta, TriangleHash> triangle_map;
};

#endif // INTERSECTABLE_MANAGER_H
