#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <assimp/scene.h>

#include "intersectables/triangle.h"
#include "material/material_loader.h"

class Model
{
public:
  Model(const std::string& path, MaterialLoader& material_loader);
  const std::vector<std::pair<Triangle, TriangleMeta>>& get_triangles() const;

private:
  void process_node(aiNode* node, const aiScene* scene);
  void process_mesh(aiMesh* mesh, const aiScene* scene);
  int load_materials(aiMaterial* material, aiTextureType type);

  MaterialLoader& material_loader;
  std::string directory;
  std::vector<std::pair<Triangle, TriangleMeta>> triangles;
};

#endif // MODEL_H
