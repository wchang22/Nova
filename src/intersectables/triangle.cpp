#include <functional>

#include "triangle.hpp"

// https://stackoverflow.com/questions/19195183/how-to-properly-hash-the-custom-struct
template <class T>
inline void hash_combine(size_t& s, const T& v) {
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

size_t TriangleHash::operator()(const Triangle& tri) const {
  size_t hash = 0;
  const glm::vec3* vertices[] = { &tri.v1, &tri.v2, &tri.v3 };

  for (int v = 0; v < 3; v++) {
    for (int i = 0; i < 3; i++) {
      hash_combine(hash, (*vertices[v])[i]);
    }
  }

  return hash;
}

std::istream& operator>>(std::istream& in, Triangle& tri) {
  in >> std::hex;
  glm::vec3* vertices[] = { &tri.v1, &tri.v2, &tri.v3 };

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
  const glm::vec3* vertices[] = { &tri.v1, &tri.v2, &tri.v3 };

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
