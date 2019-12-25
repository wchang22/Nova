#include <glm/gtx/extended_min_max.hpp>
#include <functional>
#include <algorithm>

#include "triangle.h"

const vec3 VEC_MAX(std::numeric_limits<float>::max());

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

AABB Triangle::get_clipped_bounds(const AABB& clip) const {
  auto [top, bottom] = get_bounds();

  vec3 vertices[] = { v1, v2, v3 };

  for (int axis = 0; axis < 3; axis++) {
    // Find the intersection of a triangle edge and a given plane
    auto find_intersection = [&](const vec3& a, const vec3& b, float plane) {
      vec3 d = b - a;
      assert(d[axis] != 0);
      float t = (plane - a[axis]) / d[axis];
      return a + t * d;
    };

    // Update the AABB bounds by clipping the triangle
    auto update_bounds = [&](vec3& bound, float plane,
                             vec3 (*comp)(const vec3& a, const vec3& b, const vec3& c)) {
      // Make sure the two farthest points are on either side of the plane
      if (vertices[0][axis] < plane && vertices[2][axis] > plane) {
        vec3 intrs1 = find_intersection(vertices[0], vertices[2], plane);
        vec3 intrs2;
        // Determine which side the middle point is on
        if (vertices[1][axis] < plane) {
          intrs2 = find_intersection(vertices[1], vertices[2], plane);
        } else {
          intrs2 = find_intersection(vertices[0], vertices[1], plane);
        }
        bound = comp(bound, intrs1, intrs2);
      }
    };

    vec3 plane_vertices(vertices[0][axis], vertices[1][axis], vertices[2][axis]);
    if (all(greaterThan(plane_vertices, vec3(clip.top[axis]))) ||
        all(lessThan(plane_vertices, vec3(clip.bottom[axis])))) {
      return { -VEC_MAX, VEC_MAX };
    }

    // Sort the three vertices perpendicular to the axis
    std::sort(vertices, vertices + 3, [axis](const vec3& a, const vec3& b) {
      return a[axis] < b[axis];
    });

    update_bounds(top, clip.top[axis], min);
    update_bounds(bottom, clip.bottom[axis], max);
  }
  
  return { top, bottom };
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
