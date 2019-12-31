#ifndef MODEL_H
#define MODEL_H

#include <assimp/scene.h>

#include "intersectables/intersectable_manager.h"
#include "material/material_loader.h"

class Model
{
public:
  Model(const char* path, IntersectableManager& intersectables, MaterialLoader& material_loader);

private:
  void load_model(const std::string& path);
  void process_node(aiNode* node, const aiScene* scene);
  void process_mesh(aiMesh* mesh, const aiScene* scene);
  int load_materials(aiMaterial* material, aiTextureType type);

  IntersectableManager& intersectables;
  MaterialLoader& material_loader;
  std::string directory;
};

#endif // MODEL_H
