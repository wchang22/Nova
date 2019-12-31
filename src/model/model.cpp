#include "model.h"
#include "util/exception/exception.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <cassert>

Model::Model(const char* path,
             IntersectableManager& intersectables,
             MaterialLoader& material_loader)
  : intersectables(intersectables),
    material_loader(material_loader)
{
  load_model(path);
}

void Model::load_model(const std::string& path)
{
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(path.c_str(),
                                           aiProcess_Triangulate |
                                           aiProcess_OptimizeGraph |
                                           aiProcess_OptimizeMeshes |
                                           aiProcess_ImproveCacheLocality |
                                           aiProcess_GenNormals);

  if (!scene || !scene->mRootNode || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
    throw ModelException(std::string("Assimp Error: ") + importer.GetErrorString());
  }

  directory = path.substr(0, static_cast<size_t>(path.find_last_of('/')) + 1);

  process_node(scene->mRootNode, scene);
}

void Model::process_node(aiNode* node, const aiScene* scene)
{
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
    process_mesh(mesh, scene);
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    process_node(node->mChildren[i], scene);
  }
}

void Model::process_mesh(aiMesh* mesh, const aiScene* scene)
{
  bool has_textures = mesh->mTextureCoords[0];

  std::vector<vec3> vertices, normals;
  std::vector<vec2> textures;
  vertices.reserve(mesh->mNumVertices);
  normals.reserve(mesh->mNumVertices);
  if (has_textures) {
    textures.reserve(mesh->mNumVertices);
  }

  aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
  int ambient_index = load_materials(material, aiTextureType_AMBIENT);
  int diffuse_index = load_materials(material, aiTextureType_DIFFUSE);
  int specular_index = load_materials(material, aiTextureType_SPECULAR);

  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    vertices.emplace_back(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
    normals.emplace_back(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
    if (has_textures) {
      // TODO: Support more than one texture coord
      textures.emplace_back(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
    }
  }

  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    assert(face.mNumIndices == 3);

    vec3& v1 = vertices[face.mIndices[0]];
    vec3& v2 = vertices[face.mIndices[1]];
    vec3& v3 = vertices[face.mIndices[2]];

    float lengths[] = { distance(v1, v2), distance(v2, v3), distance(v1, v3) };
    std::sort(lengths, lengths + 3);

    // Ignore degenerate triangles
    if (lengths[0] + lengths[1] <= lengths[2]) {
      continue;
    }

    vec3 n1 = normalize(normals[face.mIndices[0]]);
    vec3 n2 = normalize(normals[face.mIndices[1]]);
    vec3 n3 = normalize(normals[face.mIndices[2]]);
    vec2 t1 = has_textures ? textures[face.mIndices[0]] : vec2(0);
    vec2 t2 = has_textures ? textures[face.mIndices[1]] : vec2(0);
    vec2 t3 = has_textures ? textures[face.mIndices[2]] : vec2(0);

    intersectables.add_triangle(
      { v1, v2, v3 },
      { n1, n2, n3, t1, t2, t3, has_textures, ambient_index, diffuse_index, specular_index }
    );
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
