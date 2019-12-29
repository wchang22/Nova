#ifndef MODEL_H
#define MODEL_H

#include <assimp/scene.h>

#include "intersectables/intersectable_manager.h"

class Model
{
public:
  Model(const char* path, IntersectableManager& intersectables);

private:
  void load_model(const std::string& path);
  void process_node(aiNode* node, const aiScene* scene);
  void process_mesh(aiMesh* mesh);

  IntersectableManager& intersectables;
};

#endif // MODEL_H
