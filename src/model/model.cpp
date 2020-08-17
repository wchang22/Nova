#include "model.hpp"
#include "util/exception/exception.hpp"
#include "util/profiling/profiling.hpp"
#include "vector/vector_conversions.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>

namespace nova {

Model::Model(const std::string& path, MaterialLoader& material_loader)
  : material_loader(material_loader) {
  PROFILE_SCOPE("Load model");

  Assimp::Importer importer;

  // Normalize model to [-1, 1]
  importer.SetPropertyBool(AI_CONFIG_PP_PTV_NORMALIZE, true);

  importer.ReadFile(path.c_str(), aiProcess_Triangulate | aiProcess_ImproveCacheLocality |
                                    aiProcess_GenNormals | aiProcess_CalcTangentSpace |
                                    aiProcess_PreTransformVertices);
  const aiScene* scene =
    importer.ApplyPostProcessing(aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes);

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

  aiColor3D aikD, aikE, aikT;
  material->Get(AI_MATKEY_COLOR_DIFFUSE, aikD);
  material->Get(AI_MATKEY_COLOR_EMISSIVE, aikE);
  material->Get(AI_MATKEY_COLOR_TRANSPARENT, aikT);

  glm::vec3 kD = ai_to_glm(aikD);
  glm::vec3 kE = ai_to_glm(aikE);
  glm::vec3 kT = ai_to_glm(aikT);

  int diffuse_index = load_materials(scene, material, aiTextureType_DIFFUSE);
  int metallic_index = load_materials(scene, material, aiTextureType_METALNESS);
  int roughness_index = load_materials(scene, material, aiTextureType_DIFFUSE_ROUGHNESS);
  int normal_index = load_materials(scene, material, aiTextureType_NORMALS);

  // Some formats load normal maps into HEIGHT
  if (normal_index == -1) {
    normal_index = load_materials(scene, material, aiTextureType_HEIGHT);
  }
  // If no normal map, no need for tangents
  if (normal_index == -1) {
    has_tangents = false;
  }

  // TODO: Improve this
  // Fix some params to allow for better raytracing
  if (diffuse_index != -1 && kD == glm::vec3(0.0f)) {
    kD = glm::vec3(1.0f);
  }
  // TODO: Don't ignore transparency
  if (glm::any(glm::notEqual(kT, glm::vec3(0.0f))) && kD == glm::vec3(0.0f)) {
    kD = glm::vec3(1.0f);
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

    triangles.emplace_back<std::pair<Triangle, TriangleMeta>>({ { v1, v2, v3 }, { n1,
                                                                                  n2,
                                                                                  n3,
                                                                                  tan1,
                                                                                  tan2,
                                                                                  tan3,
                                                                                  fixed_bit1,
                                                                                  fixed_bit2,
                                                                                  fixed_bit3,
                                                                                  t1,
                                                                                  t2,
                                                                                  t3,
                                                                                  kD,
                                                                                  kE,
                                                                                  -1.0f,
                                                                                  -1.0f,
                                                                                  diffuse_index,
                                                                                  metallic_index,
                                                                                  roughness_index,
                                                                                  normal_index,
                                                                                  -1 } });
  }
}

int Model::load_materials(const aiScene* scene, aiMaterial* material, aiTextureType type) {
  uint32_t num_textures = material->GetTextureCount(type);

  if (num_textures == 0) {
    return -1;
  }

  bool srgb = type == aiTextureType_DIFFUSE;

  // TODO: Support more than one texture
  aiString path;
  material->GetTexture(type, 0, &path);

  // File path, load from filesystem
  if (path.C_Str()[0] != '*') {
    return material_loader.load_material((directory + path.C_Str()).c_str(), srgb);
  }

  // Or else, embedded texture
  int texture_index = std::stoi(path.C_Str() + 1);
  aiTexture* texture = scene->mTextures[texture_index];

  if (texture->mHeight == 0) {
    return material_loader.load_material(reinterpret_cast<const uint8_t*>(texture->pcData),
                                         texture->mWidth, srgb);
  }
  return material_loader.load_material(reinterpret_cast<const uint8_t*>(texture->pcData),
                                       texture->mWidth * texture->mHeight, srgb);
}

const std::vector<std::pair<Triangle, TriangleMeta>>& Model::get_triangles() const {
  return triangles;
}

}
