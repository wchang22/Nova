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

TEST_CASE("Clipped bounds", "[triangle]") {
  Triangle t {
    vec3(0.0f),
    vec3(1.0f, 0.0f, 1.0f),
    vec3(0.0f, 1.0f, 0.0f),
  };

  SECTION("Inside clip") {
    AABB bounds = t.get_bounds();
    AABB clip {
      vec3(2.0f),
      vec3(-2.0f),
    };

    REQUIRE(t.get_clipped_bounds(clip) == bounds);
  }

  SECTION("Clip inside") {
    AABB clip {
      vec3(0.9f),
      vec3(0.0f),
    };

    REQUIRE(t.get_clipped_bounds(clip) == clip);
  }

  SECTION("No bounds") {
    AABB clip {
      vec3(-0.1f),
      vec3(-1.0f),
    };
    AABB bounds {
      vec3(-std::numeric_limits<float>::max()),
      vec3(std::numeric_limits<float>::max())
    };

    REQUIRE(t.get_clipped_bounds(clip) == bounds);
  }

  SECTION("Clip 1") {
    AABB clip {
      vec3(0.5f),
      vec3(-0.5f),
    };
    AABB bounds {
      vec3(0.5f),
      vec3(0.0f),
    };

    REQUIRE(t.get_clipped_bounds(clip) == bounds);
  }

  SECTION("Clip 2") {
    AABB clip {
      vec3(1.2f, 0.5f, 1.0f),
      vec3(0.75f, -0.5f, -1.0f),
    };
    AABB bounds {
      vec3(1.0f, 0.25f, 1.0f),
      vec3(0.75f, 0.0f, 0.75f),
    };

    REQUIRE(t.get_clipped_bounds(clip) == bounds);
  }

  SECTION("Clip 3") {
    AABB clip {
      vec3(0.5f, 1.2f, 1.0f),
      vec3(-0.5f, -0.5f, -1.0f),
    };
    AABB bounds {
      vec3(0.5f, 1.0f, 0.5f),
      vec3(0.0f, 0.0f, 0.0f),
    };

    REQUIRE(t.get_clipped_bounds(clip) == bounds);
  }
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