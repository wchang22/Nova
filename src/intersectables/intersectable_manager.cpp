#include "intersectable_manager.hpp"
#include "bvh/bvh.hpp"
#include "constants.hpp"
#include "util/exception/exception.hpp"
#include "vector/vector_conversions.hpp"

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

void IntersectableManager::add_light(const AreaLight& light) {
  const auto& [intensity, position, normal, dims] = light;
  glm::vec3 v = normal;
  glm::vec3 aux = std::abs(v.y) > 0.1 ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec3 u = glm::normalize(glm::cross(v, aux));
  glm::vec3 w = glm::cross(u, v);

  Triangle tri1 {
    position - dims.x * u - dims.y * w,
    position - dims.x * u + dims.y * w,
    position + dims.x * u + dims.y * w,
  };
  Triangle tri2 {
    position + dims.x * u + dims.y * w,
    position + dims.x * u - dims.y * w,
    position - dims.x * u - dims.y * w,
  };
  TriangleMeta meta { light.normal,
                      light.normal,
                      light.normal,
                      {},
                      {},
                      {},
                      {},
                      {},
                      {},
                      {},
                      {},
                      {},
                      glm::vec3(1.0f),
                      light.intensity,
                      -1,
                      -1,
                      -1,
                      -1,
                      static_cast<int>(lights.size()) };
  add_triangle(tri1, meta);
  add_triangle(tri2, meta);
  lights.push_back(light);
}

void IntersectableManager::clear() {
  triangles.clear();
  triangle_map.clear();
  lights.clear();
}

IntersectableData IntersectableManager::build() {
  BVH bvh(triangles);
  std::vector<FlatBVHNode> bvh_data = bvh.build();

  // BVH modifies the order of triangles, so we need to look up the meta data
  // Separate triangle normals from triangle data, as we do not need the normal during
  // intersection, and this reduces cache pressure
  std::vector<TriangleData> triangle_data;
  std::vector<TriangleMetaData> triangle_meta_data;
  std::vector<AreaLightData> light_data;
  triangle_data.reserve(triangles.size());
  triangle_meta_data.reserve(triangles.size());
  light_data.reserve(lights.size());

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
    transform = glm::transpose(glm::inverse(transform));

    triangle_data.push_back(
      { glm_to_float4(transform[0]), glm_to_float4(transform[1]), glm_to_float4(transform[2]) });

    const auto& meta = triangle_map[tri];
    triangle_meta_data.push_back({
      glm_to_float3(meta.normal1),
      glm_to_float3(meta.normal2),
      glm_to_float3(meta.normal3),
      glm_to_float3(meta.tangent1),
      glm_to_float3(meta.tangent2),
      glm_to_float3(meta.tangent3),
      glm_to_float3(meta.bitangent1),
      glm_to_float3(meta.bitangent2),
      glm_to_float3(meta.bitangent3),
      glm_to_float2(meta.texture_coord1),
      glm_to_float2(meta.texture_coord2),
      glm_to_float2(meta.texture_coord3),
      glm_to_float3(meta.kD),
      glm_to_float3(meta.kE),
      meta.diffuse_index,
      meta.metallic_index,
      meta.roughness_index,
      meta.normal_index,
      meta.light_index,
    });
  }

  std::transform(lights.begin(), lights.end(), std::back_inserter(light_data),
                 [](const auto& light) -> AreaLightData {
                   return {
                     glm_to_float3(light.intensity),
                     glm_to_float3(light.position),
                     glm_to_float3(light.normal),
                     glm_to_float2(light.dims),
                   };
                 });

  return { triangle_data, triangle_meta_data, bvh_data, light_data };
}

}
