#include "intersectable_manager.h"
#include "acceleration/bvh.h"

void IntersectableManager::add_triangle(const Triangle& tri, const Material& mat) {
  triangles.push_back(tri);
  triangle_map[tri] = mat;
}

void IntersectableManager::build_buffers(const cl::Context& context,
                                         std::pair<cl::Buffer, size_t>& triangle_buf,
                                         cl::Buffer& materials_buf,
                                         cl::Buffer& bvh_buf) {
  BVH bvh(triangles);
  bvh_buf = bvh.build_bvh_buffer(context);

  // bvh modifies the order of triangles, so we need to look up the material
  std::vector<TriangleData> triangle_data;
  std::vector<MaterialData> material_data;
  triangle_data.reserve(triangles.size());
  material_data.reserve(triangles.size());

  for (const auto& tri : triangles) {
    vec3 vertex = tri.v1;
    vec3 edge1 = tri.v2 - tri.v1;
    vec3 edge2 = tri.v3 - tri.v1;
    vec3 normal = cross(edge1, edge2);

    Material mat = triangle_map[tri];

    triangle_data.push_back({
      { {vertex.x, vertex.y, vertex.z} },
      { {normal.x, normal.y, normal.z} },
      { {edge1.x, edge1.y, edge1.z} },
      { {edge2.x, edge2.y, edge2.z} }
    });
    material_data.push_back({
      { {mat.ambient.x, mat.ambient.y, mat.ambient.z} },
      { {mat.diffuse.x, mat.diffuse.y, mat.diffuse.z} },
      { {mat.specular.x, mat.specular.y, mat.specular.z} },
    });
  }

  cl::Buffer tri_buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     triangle_data.size() * sizeof(decltype(triangle_data)::value_type),
                     triangle_data.data());
  triangle_buf = { tri_buf, triangle_data.size() };
  cl::Buffer mat_buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     material_data.size() * sizeof(decltype(material_data)::value_type),
                     material_data.data());
  materials_buf = mat_buf;
}
