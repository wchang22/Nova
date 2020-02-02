#include <catch.hpp>
#include <sstream>

#include "intersectables/triangle.h"

TEST_CASE("Triangle bounds", "[triangle]") {
  Triangle t1 { 
    vec3(1.32f, 3.56f, -4.12f),
    vec3(0.0f),
    vec3(-5.62f, 9.83f, -2.21f),
  };

  AABB expected_bounds {
    vec3(1.32f, 9.83f, 0.0f),
    vec3(-5.62f, 0.0f, -4.12f),
  };

  REQUIRE(t1.get_bounds() == expected_bounds);
}

TEST_CASE("(De)serialize triangle", "[triangle]") {
  Triangle t1 { 
    vec3(1.32f, 3.56f, -4.12f),
    vec3(0.0f),
    vec3(-5.62f, 9.83f, -2.21f),
  };
  Triangle t2;

  std::stringstream ss;
  ss << t1;
  ss >> t2;

  REQUIRE(t1 == t2);
}

TEST_CASE("Hash triangle", "[triangle]") {
  Triangle t1 { 
    vec3(1.32f, 3.56f, -4.12f),
    vec3(0.0f),
    vec3(-5.62f, 9.83f, -2.21f),
  };
  Triangle t2 { 
    vec3(1.321f, 3.56f, -4.12f),
    vec3(0.0f),
    vec3(-5.62f, 9.83f, -2.21f),
  };

  TriangleHash h;

  REQUIRE(h(t1) != h(t2));
  REQUIRE(h(t1) == h(t1));
  REQUIRE(h(t2) == h(t2));
}