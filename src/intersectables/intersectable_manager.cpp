#include "intersectable_manager.h"
#include "acceleration/bvh.h"
#include "util/exception/exception.h"
#include "constants.h"

IntersectableManager::IntersectableManager(const std::string& name) : name(name) {}

void IntersectableManager::add_triangle(const Triangle& tri, const TriangleMeta& meta) {
  if (triangles.size() >= MAX_TRIANGLES) {
    throw TriangleException("Max number of triangles exceeded");
  }

  triangles.push_back(tri);
  triangle_map[tri] = meta;
}

void IntersectableManager::add_model(const Model& model) {
  for (const auto& [tri, meta] : model.get_triangles()) {
    add_triangle(tri, meta);
  }
}

void IntersectableManager::build_buffers(const cl::Context& context,
                                         cl::Buffer& triangle_buf,
                                         cl::Buffer& tri_meta_buf,
                                         cl::Buffer& bvh_buf) {
  BVH bvh(name, triangles);
  bvh_buf = bvh.build_bvh_buffer(context);

  // BVH modifies the order of triangles, so we need to look up the meta data
  // Separate triangle normals from triangle data, as we do not need the normal during intersection,
  // and this reduces cache pressure
  std::vector<TriangleData> triangle_data;
  std::vector<TriangleMetaData> meta_data;
  triangle_data.reserve(triangles.size());
  meta_data.reserve(triangles.size());

  for (const auto& tri : triangles) {
    const auto& [v1, v2, v3] = tri;
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

    const auto& meta = triangle_map[tri];
    meta_data.push_back({
      { {meta.normal1.x, meta.normal1.y, meta.normal1.z} },
      { {meta.normal2.x, meta.normal2.y, meta.normal2.z} },
      { {meta.normal3.x, meta.normal3.y, meta.normal3.z} },
      { {meta.tangent1.x, meta.tangent1.y, meta.tangent1.z} },
      { {meta.tangent2.x, meta.tangent2.y, meta.tangent2.z} },
      { {meta.tangent3.x, meta.tangent3.y, meta.tangent3.z} },
      { {meta.bitangent1.x, meta.bitangent1.y, meta.bitangent1.z} },
      { {meta.bitangent2.x, meta.bitangent2.y, meta.bitangent2.z} },
      { {meta.bitangent3.x, meta.bitangent3.y, meta.bitangent3.z} },
      { {meta.texture_coord1.x, meta.texture_coord1.y} },
      { {meta.texture_coord2.x, meta.texture_coord2.y} },
      { {meta.texture_coord3.x, meta.texture_coord3.y} },
      meta.diffuse_index,
      meta.metallic_index,
      meta.roughness_index,
      meta.ambient_occlusion_index,
      meta.normal_index,
    });
  }

  triangle_buf = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     triangle_data.size() * sizeof(decltype(triangle_data)::value_type),
                     triangle_data.data());
  tri_meta_buf = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     meta_data.size() * sizeof(decltype(meta_data)::value_type),
                     meta_data.data());
}
