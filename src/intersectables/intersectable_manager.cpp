#include "intersectable_manager.hpp"
#include "bvh/bvh.hpp"
#include "constants.hpp"
#include "util/exception/exception.hpp"

namespace nova {

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

void IntersectableManager::clear() {
  triangles.clear();
  triangle_map.clear();
}

IntersectableData IntersectableManager::build() {
  BVH bvh(triangles);
  std::vector<FlatBVHNode> bvh_data = bvh.build();

  // BVH modifies the order of triangles, so we need to look up the meta data
  // Separate triangle normals from triangle data, as we do not need the normal during
  // intersection, and this reduces cache pressure
  std::vector<TriangleData> triangle_data;
  std::vector<TriangleMetaData> triangle_meta_data;
  triangle_data.reserve(triangles.size());
  triangle_meta_data.reserve(triangles.size());

  for (const auto& tri : triangles) {
    const auto& [v1, v2, v3] = tri;
    glm::vec3 e1 = v2 - v1;
    glm::vec3 e2 = v3 - v1;
    glm::vec3 normal = glm::normalize(glm::cross(e1, e2));

    // Create woop transformation matrix to transform ray to unit triangle space
    // http://www.sven-woop.de/papers/2004-GH-SaarCOR.pdf
    glm::mat4 transform;
    transform[0] = glm::vec4(e1, 0);
    transform[1] = glm::vec4(e2, 0);
    transform[2] = glm::vec4(normal - v1, 0);
    transform[3] = glm::vec4(v1, 1);
    transform = glm::inverse(transform);

    triangle_data.push_back(
      { { { transform[0][0], transform[1][0], transform[2][0], transform[3][0] },
          { transform[0][1], transform[1][1], transform[2][1], transform[3][1] },
          { transform[0][2], transform[1][2], transform[2][2], transform[3][2] } } });

    const auto& meta = triangle_map[tri];
    triangle_meta_data.push_back({
      { meta.normal1.x, meta.normal1.y, meta.normal1.z },
      { meta.normal2.x, meta.normal2.y, meta.normal2.z },
      { meta.normal3.x, meta.normal3.y, meta.normal3.z },
      { meta.tangent1.x, meta.tangent1.y, meta.tangent1.z },
      { meta.tangent2.x, meta.tangent2.y, meta.tangent2.z },
      { meta.tangent3.x, meta.tangent3.y, meta.tangent3.z },
      { meta.bitangent1.x, meta.bitangent1.y, meta.bitangent1.z },
      { meta.bitangent2.x, meta.bitangent2.y, meta.bitangent2.z },
      { meta.bitangent3.x, meta.bitangent3.y, meta.bitangent3.z },
      { meta.texture_coord1.x, meta.texture_coord1.y },
      { meta.texture_coord2.x, meta.texture_coord2.y },
      { meta.texture_coord3.x, meta.texture_coord3.y },
      meta.diffuse_index,
      meta.metallic_index,
      meta.roughness_index,
      meta.ambient_occlusion_index,
      meta.normal_index,
    });
  }

  return { triangle_data, triangle_meta_data, bvh_data };
}

}
