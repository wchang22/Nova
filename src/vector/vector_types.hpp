#ifndef VECTOR_TYPES_HPP
#define VECTOR_TYPES_HPP

#include <array>

namespace nova {
using vec2f = std::array<float, 2>;
using vec3f = std::array<float, 3>;
using vec4f = std::array<float, 4>;
using vec2i = std::array<int32_t, 2>;
using vec3i = std::array<int32_t, 3>;
using vec4i = std::array<int32_t, 4>;
using vec2u = std::array<uint32_t, 2>;
using vec3u = std::array<uint32_t, 3>;
using vec4u = std::array<uint32_t, 4>;
using vec2u8 = std::array<uint8_t, 2>;
using vec3u8 = std::array<uint8_t, 3>;
using vec4u8 = std::array<uint8_t, 4>;
}

#endif // VECTOR_TYPES_HPP