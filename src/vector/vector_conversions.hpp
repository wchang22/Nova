#ifndef VECTOR_CONVERSIONS_HPP
#define VECTOR_CONVERSIONS_HPP

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/types.h>

#include "backend/types.hpp"
#include "vector/vector_types.hpp"

namespace nova {

template <typename T, size_t N>
inline glm::vec<N, T, glm::defaultp> vec_to_glm(const std::array<T, N>& in) {
  glm::vec<N, T, glm::defaultp> out;
  std::copy(in.data(), in.data() + N, glm::value_ptr(out));
  return out;
}

template <typename T, int N>
inline std::array<T, N> glm_to_vec(const glm::vec<N, T, glm::defaultp>& in) {
  std::array<T, N> out;
  std::copy(glm::value_ptr(in), glm::value_ptr(in) + N, out.data());
  return out;
}

inline float3 vec_to_float3(const vec3f& in) { return { in[0], in[1], in[2] }; }

inline float4 glm_to_float4(const glm::vec4& in) { return { in.x, in.y, in.z, in.w }; }

inline float3 glm_to_float3(const glm::vec3& in) { return { in.x, in.y, in.z }; }

inline float2 glm_to_float2(const glm::vec2& in) { return { in.x, in.y }; }

inline aiVector2D make_aiVector2D(const aiVector3D& in) { return { in.x, in.y }; }

inline float3 ai_to_float3(const aiColor3D& in) { return { in.r, in.g, in.b }; }

inline float3 ai_to_float3(const aiVector3D& in) { return { in.x, in.y, in.z }; }

inline glm::vec3 ai_to_glm(const aiColor3D& in) { return { in.r, in.g, in.b }; }

inline glm::vec2 ai_to_glm(const aiVector2D& in) { return { in.x, in.y }; }

inline glm::vec3 ai_to_glm(const aiVector3D& in) { return { in.x, in.y, in.z }; }


}

#endif // VECTOR_CONVERSIONS_HPP