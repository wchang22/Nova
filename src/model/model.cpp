#include "model.h"
#include "util/exception/exception.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <cassert>

Model::Model(const char* path, IntersectableManager& intersectables)
  : intersectables(intersectables)
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
                                           aiProcess_ImproveCacheLocality);

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
    process_mesh(mesh);
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    process_node(node->mChildren[i], scene);
  }
}

void Model::process_mesh(aiMesh* mesh)
{
  std::vector<vec3> vertices;
  vertices.reserve(mesh->mNumVertices);
  std::vector<unsigned int> indices(mesh->mNumVertices);

  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    vertices.emplace_back(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
  }

  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    assert(face.mNumIndices == 3);

    intersectables.add_triangle({
      vertices[face.mIndices[0]], vertices[face.mIndices[1]], vertices[face.mIndices[2]]
    }, {
      vec3(0.1, 0.1, 0.1), vec3(0.4, 0.4, 0.4), vec3(0.7, 0.7, 0.7)
    });
  }
}
