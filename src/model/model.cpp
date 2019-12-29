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
                                           aiProcess_ImproveCacheLocality |
                                           aiProcess_GenNormals);

  if (!scene || !scene->mRootNode || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
    throw ModelException(std::string("Assimp Error: ") + importer.GetErrorString());
  }

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
  std::vector<vec3> vertices, normals;
  vertices.reserve(mesh->mNumVertices);
  normals.reserve(mesh->mNumVertices);

  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    vertices.emplace_back(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
    normals.emplace_back(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
  }

  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    assert(face.mNumIndices == 3);

    vec3& v1 = vertices[face.mIndices[0]];
    vec3& v2 = vertices[face.mIndices[1]];
    vec3& v3 = vertices[face.mIndices[2]];
    vec3 n1 = normalize(normals[face.mIndices[0]]);
    vec3 n2 = normalize(normals[face.mIndices[1]]);
    vec3 n3 = normalize(normals[face.mIndices[2]]);

    float lengths[] = { distance(v1, v2), distance(v2, v3), distance(v1, v3) };
    std::sort(lengths, lengths + 3);

    // Ignore degenerate triangles
    if (lengths[0] + lengths[1] <= lengths[2]) {
      continue;
    }

    intersectables.add_triangle({ v1, v2, v3 }, { n1, n2, n3 }, {
      vec3(0.1, 0.1, 0.1), vec3(0.4, 0.4, 0.4), vec3(0.7, 0.7, 0.7)
    });
  }
}
