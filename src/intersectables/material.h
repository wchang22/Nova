#ifndef MATERIAL_H
#define MATERIAL_H

#include <CL/cl2.hpp>
#include <glm/glm.hpp>

using namespace glm;

struct Material {
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

struct MaterialData {
  cl_float3 ambient;
  cl_float3 diffuse;
  cl_float3 specular;
};

#endif // MATERIAL_H