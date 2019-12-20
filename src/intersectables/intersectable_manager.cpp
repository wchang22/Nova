#include "intersectable_manager.h"

void IntersectableManager::add_triangle(const Triangle& tri, const Material& mat) {
  vec3 vertex = tri.v1;
  vec3 edge1 = tri.v2 - tri.v1;
  vec3 edge2 = tri.v3 - tri.v1;
  vec3 normal = cross(edge1, edge2);

  triangles.emplace_back<TriangleData, MaterialData>({
    { {vertex.x, vertex.y, vertex.z} },
    { {normal.x, normal.y, normal.z} },
    { {edge1.x, edge1.y, edge1.z} },
    { {edge2.x, edge2.y, edge2.z} }
  }, {
    { {mat.ambient.x, mat.ambient.y, mat.ambient.z} },
    { {mat.diffuse.x, mat.diffuse.y, mat.diffuse.z} },
    { {mat.specular.x, mat.specular.y, mat.specular.z} },
  });
}

std::pair<cl::Buffer, size_t> IntersectableManager::build_buffer(const cl::Context& context) {
  cl::Buffer buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                 triangles.size() * sizeof(decltype(triangles)::value_type), triangles.data());
  return { buf, triangles.size() };
}
