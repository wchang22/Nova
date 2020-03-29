#include "model.hpp"
#include "util/exception/exception.hpp"
#include "util/profiling/profiling.hpp"
#include "vector/vector_conversions.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <cassert>
#include <glm/glm.hpp>

namespace nova {

Model::Model(const std::string& path, MaterialLoader& material_loader)
  : material_loader(material_loader) {
  PROFILE_SCOPE("Load model");

  Assimp::Importer importer;
  const aiScene* scene =
    importer.ReadFile(path.c_str(), aiProcess_Triangulate | aiProcess_OptimizeGraph |
                                      aiProcess_OptimizeMeshes | aiProcess_ImproveCacheLocality |
                                      aiProcess_GenNormals | aiProcess_CalcTangentSpace);

  if (!scene || !scene->mRootNode || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
    throw ModelException(std::string("Assimp Error: ") + importer.GetErrorString());
  }

  directory = path.substr(0, static_cast<size_t>(path.find_last_of('/')) + 1);

  process_node(scene->mRootNode, scene);
}

void Model::process_node(aiNode* node, const aiScene* scene) {
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
    process_mesh(mesh, scene);
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    process_node(node->mChildren[i], scene);
  }
}

void Model::process_mesh(aiMesh* mesh, const aiScene* scene) {
  bool has_textures = mesh->mTextureCoords[0];
  bool has_tangents = mesh->mTangents;

  std::vector<glm::vec3> vertices, normals, tangents, bitangents;
  std::vector<glm::vec2> textures;
  vertices.reserve(mesh->mNumVertices);
  normals.reserve(mesh->mNumVertices);
  if (has_textures) {
    textures.reserve(mesh->mNumVertices);
  }
  if (has_tangents) {
    tangents.reserve(mesh->mNumVertices);
    bitangents.reserve(mesh->mNumVertices);
  }

  aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
  int diffuse_index = load_materials(material, aiTextureType_DIFFUSE);
  int metallic_index = load_materials(material, aiTextureType_METALNESS);
  int roughness_index = load_materials(material, aiTextureType_DIFFUSE_ROUGHNESS);
  int ambient_occlusion_index = load_materials(material, aiTextureType_AMBIENT_OCCLUSION);
  int normal_index = load_materials(material, aiTextureType_NORMALS);
  // Some formats load normal maps into HEIGHT
  if (normal_index == -1) {
    normal_index = load_materials(material, aiTextureType_HEIGHT);
  }
  // If no normal map, no need for tangents
  if (normal_index == -1) {
    has_tangents = false;
  }

  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    vertices.emplace_back(ai_to_glm(mesh->mVertices[i]));
    normals.emplace_back(ai_to_glm(mesh->mNormals[i]));
    if (has_textures) {
      // TODO: Support more than one texture coord
      textures.emplace_back(ai_to_glm(make_aiVector2D(mesh->mTextureCoords[0][i])));
    }
    if (has_tangents) {
      tangents.emplace_back(ai_to_glm(mesh->mTangents[i]));
      bitangents.emplace_back(ai_to_glm(mesh->mBitangents[i]));
    }
  }

  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    assert(face.mNumIndices == 3);

    glm::vec3& v1 = vertices[face.mIndices[0]];
    glm::vec3& v2 = vertices[face.mIndices[1]];
    glm::vec3& v3 = vertices[face.mIndices[2]];

    float lengths[] = { glm::distance(v1, v2), glm::distance(v2, v3), glm::distance(v1, v3) };
    std::sort(lengths, lengths + 3);

    // Ignore degenerate triangles
    if (lengths[0] + lengths[1] <= lengths[2]) {
      continue;
    }

    glm::vec3 n1 = glm::normalize(normals[face.mIndices[0]]);
    glm::vec3 n2 = glm::normalize(normals[face.mIndices[1]]);
    glm::vec3 n3 = glm::normalize(normals[face.mIndices[2]]);
    glm::vec3 tan1 = has_tangents ? tangents[face.mIndices[0]] : glm::vec3(0);
    glm::vec3 tan2 = has_tangents ? tangents[face.mIndices[1]] : glm::vec3(0);
    glm::vec3 tan3 = has_tangents ? tangents[face.mIndices[2]] : glm::vec3(0);
    glm::vec3 bit1 = has_tangents ? bitangents[face.mIndices[0]] : glm::vec3(0);
    glm::vec3 bit2 = has_tangents ? bitangents[face.mIndices[1]] : glm::vec3(0);
    glm::vec3 bit3 = has_tangents ? bitangents[face.mIndices[2]] : glm::vec3(0);
    glm::vec2 t1 = has_textures ? textures[face.mIndices[0]] : glm::vec2(0);
    glm::vec2 t2 = has_textures ? textures[face.mIndices[1]] : glm::vec2(0);
    glm::vec2 t3 = has_textures ? textures[face.mIndices[2]] : glm::vec2(0);

    // Fixes models with symmetric uv coordinates
    glm::vec3 fixed_bit1 = glm::cross(n1, tan1);
    glm::vec3 fixed_bit2 = glm::cross(n2, tan2);
    glm::vec3 fixed_bit3 = glm::cross(n3, tan3);
    if (glm::dot(fixed_bit1, bit1) < 0.0) {
      fixed_bit1 *= -1.0f;
    }
    if (glm::dot(fixed_bit2, bit2) < 0.0) {
      fixed_bit2 *= -1.0f;
    }
    if (glm::dot(fixed_bit3, bit3) < 0.0) {
      fixed_bit3 *= -1.0f;
    }

    triangles.emplace_back<std::pair<Triangle, TriangleMeta>>(
      { { v1, v2, v3 },
        { n1, n2, n3, tan1, tan2, tan3, fixed_bit1, fixed_bit2, fixed_bit3, t1, t2, t3,
          diffuse_index, metallic_index, roughness_index, ambient_occlusion_index,
          normal_index } });
  }
}

int Model::load_materials(aiMaterial* material, aiTextureType type) {
  uint32_t num_textures = material->GetTextureCount(type);

  if (num_textures == 0) {
    return -1;
  }

  // TODO: Support more than one texture
  aiString path;
  material->GetTexture(type, 0, &path);
  return material_loader.load_material((directory + path.C_Str()).c_str());
}

const std::vector<std::pair<Triangle, TriangleMeta>>& Model::get_triangles() const {
  return triangles;
}

}
