#include "intersectable_manager.h"
#include "acceleration/bvh.h"

void IntersectableManager::add_triangle(const Triangle& tri, const Material& mat) {
  triangles.push_back(tri);
  triangle_map[tri] = mat;
}

void IntersectableManager::build_buffers(const cl::Context& context,
                                         std::pair<cl::Buffer, size_t>& triangle_buf,
                                         cl::Buffer& bvh_buf) {
  BVH bvh(triangles);
  bvh_buf = bvh.build_bvh_buffer(context);

  // bvh modifies the order of triangles, so we need to look up the material
  std::vector<std::pair<TriangleData, MaterialData>> triangle_data;

  for (const auto& tri : triangles) {
    vec3 vertex = tri.v1;
    vec3 edge1 = tri.v2 - tri.v1;
    vec3 edge2 = tri.v3 - tri.v1;
    vec3 normal = cross(edge1, edge2);

    Material mat = triangle_map[tri];

    triangle_data.emplace_back<TriangleData, MaterialData>({
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

  cl::Buffer buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                 triangle_data.size() * sizeof(decltype(triangle_data)::value_type),
                 triangle_data.data());
  triangle_buf = { buf, triangle_data.size() };
}
