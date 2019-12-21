#include <glm/gtx/extended_min_max.hpp>

#include "triangle.h"

// https://stackoverflow.com/questions/19195183/how-to-properly-hash-the-custom-struct
template <class T>
inline void hash_combine(size_t& s, const T & v)
{
  std::hash<T> h;
  s^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2);
}

size_t TriangleHash::operator()(const Triangle& tri) const {
  size_t hash = 0;

  for (int i = 0; i < 3; i++) {
    hash_combine(hash, tri.v1[i]);
  }
  for (int i = 0; i < 3; i++) {
    hash_combine(hash, tri.v2[i]);
  }
  for (int i = 0; i < 3; i++) {
    hash_combine(hash, tri.v3[i]);
  }

  return hash;
}

std::pair<vec3, vec3> Triangle::get_bounds() const {
  vec3 top = max(v1, v2, v3);
  vec3 bottom = min(v1, v2, v3);
  
  return { top, bottom };
}

bool Triangle::operator==(const Triangle& t) const {
  return t.v1 == v1 && t.v2 == v2 && t.v3 == v3;
}
