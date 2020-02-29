#include <glm/gtx/extended_min_max.hpp>
#include <functional>
#include <algorithm>
#include <glm/gtx/string_cast.hpp>
#include <numeric>
#include <iostream>
#include <cassert>

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
  const vec3* vertices[] = { &tri.v1, &tri.v2, &tri.v3 };

  for (int v = 0; v < 3; v++) {
    for (int i = 0; i < 3; i++) {
      hash_combine(hash, (*vertices[v])[i]);
    }
  }

  return hash;
}

AABB Triangle::get_bounds() const {
  vec3 top = max(v1, v2, v3);
  vec3 bottom = min(v1, v2, v3);

  return { top, bottom };
}

void Triangle::split_aabb(AABB& left, AABB& right, const AABB& aabb, int axis, float split) const {
  constexpr auto lerp = [](const vec3& a, const vec3& b, float t) {
    return a + t * (b - a);
  };

  left = NO_INTERSECTION;
  right = NO_INTERSECTION;

  const vec3* vertices[] = { &v1, &v2, &v3 };
  const vec3* v1 = vertices[2];

  for (int i = 0; i < 3; i++) {
    const vec3* v0 = v1;
    v1 = vertices[i];
    float v0p = (*v0)[axis];
    float v1p = (*v1)[axis];

    if (v0p <= split) {
      left.grow(*v0);
    }
    if (v0p >= split) {
      right.grow(*v0);
    }

    if ((v0p < split && v1p > split) || (v0p > split && v1p < split)) {
      vec3 t = lerp(*v0, *v1, std::clamp((split - v0p) / (v1p - v0p), 0.0f, 1.0f));
      left.grow(t);
      right.grow(t);
    }
  }

  left.top[axis] = split;
  right.bottom[axis] = split;
  left.shrink(aabb);
  right.shrink(aabb);
}

bool Triangle::operator==(const Triangle& t) const {
  return t.v1 == v1 && t.v2 == v2 && t.v3 == v3;
}

std::istream& operator>>(std::istream& in, Triangle& tri) {
  in >> std::hex;
  vec3* vertices[] = { &tri.v1, &tri.v2, &tri.v3 };

  for (int v = 0; v < 3; v++) {
    for (int i = 0; i < 3; i++) {
      uint32_t x;
      float* f = reinterpret_cast<float*>(&x);
      in >> x;
      (*vertices[v])[i] = *f;
    }
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const Triangle& tri) {
  out << std::hex;
  const vec3* vertices[] = { &tri.v1, &tri.v2, &tri.v3 };

  for (int v = 0; v < 3; v++) {
    for (int i = 0; i < 3; i++) {
      // Store floating point as an integer, so we don't lose precision
      float f = (*vertices[v])[i];
      uint32_t* x = reinterpret_cast<uint32_t*>(&f);
      out << *x << " ";
    }
  }
  return out;
}
