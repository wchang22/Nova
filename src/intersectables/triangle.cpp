#include <glm/gtx/extended_min_max.hpp>
#include <functional>
#include <algorithm>
#include <glm/gtx/string_cast.hpp>
#include <numeric>
#include <iostream>
#include <cassert>

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
  AABB bounds = get_bounds();

  if (!bounds.intersects(clip) && !clip.intersects(bounds)) {
    return { -VEC_MAX, VEC_MAX };
  }
  if (bounds.is_in(clip)) {
    return bounds;
  }
  if (clip.is_in(bounds)) {
    return clip;
  }

  std::vector<vec3> output_vertices { v1, v2, v3 };

  for (int axis = 0; axis < 3; axis++) {
    auto find_intersection = [&](const vec3& a, const vec3& b, float plane) {
      vec3 d = b - a;
      assert(d[axis] != 0);
      float t = (plane - a[axis]) / d[axis];
      return a + t * d;
    };

    auto plane_clip = [&](float plane, bool (*is_inside)(float, float)) {
      if (output_vertices.empty()) {
        return;
      }

      std::vector<vec3> input_vertices;
      std::swap(input_vertices, output_vertices);

      vec3 prev_point = input_vertices.back();
      for (const auto& curr_point : input_vertices) {
        if (is_inside(curr_point[axis], plane)) {
          if (!is_inside(prev_point[axis], plane)) {
            output_vertices.push_back(find_intersection(prev_point, curr_point, plane));
          }
          output_vertices.push_back(curr_point);
        } else if (is_inside(prev_point[axis], plane)) {
          output_vertices.push_back(find_intersection(prev_point, curr_point, plane));
        }

        prev_point = curr_point;
      }
    };

    plane_clip(clip.bottom[axis], [](float point, float plane) { return point >= plane; });
    plane_clip(clip.top[axis], [](float point, float plane) { return point <= plane; });
  }

  if (output_vertices.empty()) {
    return { -VEC_MAX, VEC_MAX };
  }

  assert(output_vertices.size() >= 3);

  vec3 top = std::accumulate(output_vertices.begin(), output_vertices.end(),
                             -VEC_MAX, [](const vec3& a, const vec3& b) {
                               return max(a, b);
                             });
  vec3 bottom = std::accumulate(output_vertices.begin(), output_vertices.end(),
                                VEC_MAX, [](const vec3& a, const vec3& b) {
                                  return min(a, b);
                                });
  top = clamp(top, clip.bottom, clip.top);
  bottom = clamp(bottom, clip.bottom, clip.top);

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
