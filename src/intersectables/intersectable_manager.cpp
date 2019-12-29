#include "intersectable_manager.h"
#include "acceleration/bvh.h"

#include <glm/gtx/string_cast.hpp>

void IntersectableManager::add_triangle(const Triangle& tri, const TriangleMeta& meta,
                                        const Material& mat) {
  triangles.push_back(tri);
  triangle_map[tri] = { meta, mat };
}

void IntersectableManager::build_buffers(const cl::Context& context,
                                         cl::Buffer& triangle_buf,
                                         cl::Buffer& tri_meta_buf,
                                         cl::Buffer& materials_buf,
                                         cl::Buffer& bvh_buf) {
  BVH bvh(triangles);
  bvh_buf = bvh.build_bvh_buffer(context);

  // BVH modifies the order of triangles, so we need to look up the material
  // Separate triangle normals from triangle data, as we do not need the normal during intersection,
  // and this reduces cache pressure
  std::vector<TriangleData> triangle_data;
  std::vector<TriangleMetaData> meta_data;
  std::vector<MaterialData> material_data;
  triangle_data.reserve(triangles.size());
  material_data.reserve(triangles.size());

  for (const auto& tri : triangles) {
    auto& [v1, v2, v3] = tri;
    vec3 e1 = v2 - v1;
    vec3 e2 = v3 - v1;
    vec3 normal = normalize(cross(e1, e2));

    // Create woop transformation matrix to transform ray to unit triangle space
    // http://www.sven-woop.de/papers/2004-GH-SaarCOR.pdf
    mat4 transform;
    transform[0] = vec4(e1, 0);
    transform[1] = vec4(e2, 0);
    transform[2] = vec4(normal - v1, 0);
    transform[3] = vec4(v1, 1);
    transform = inverse(transform);

    triangle_data.push_back({
      { {transform[0][0], transform[1][0], transform[2][0], transform[3][0]} },
      { {transform[0][1], transform[1][1], transform[2][1], transform[3][1]} },
      { {transform[0][2], transform[1][2], transform[2][2], transform[3][2]} }
    });

    auto [meta, mat] = triangle_map[tri];
    material_data.push_back({
      { {mat.ambient.x, mat.ambient.y, mat.ambient.z} },
      { {mat.diffuse.x, mat.diffuse.y, mat.diffuse.z} },
      { {mat.specular.x, mat.specular.y, mat.specular.z} },
    });
    meta_data.push_back({
      { {meta.normal1.x, meta.normal1.y, meta.normal1.z} },
      { {meta.normal2.x, meta.normal2.y, meta.normal2.z} },
      { {meta.normal3.x, meta.normal3.y, meta.normal3.z} },
    });
  }

  triangle_buf = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     triangle_data.size() * sizeof(decltype(triangle_data)::value_type),
                     triangle_data.data());
  tri_meta_buf = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     meta_data.size() * sizeof(decltype(meta_data)::value_type),
                     meta_data.data());
  materials_buf = cl::Buffer (context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     material_data.size() * sizeof(decltype(material_data)::value_type),
                     material_data.data());
}
