#include <catch.hpp>

#include "intersectables/aabb.h"

TEST_CASE("AABB surface area", "[aabb]") {
  AABB aabb1 {
    vec3(1.0f),
    vec3(0.0f),
  };

  SECTION("AABB calculate SA") {
    REQUIRE(aabb1.get_surface_area() == Approx(6.0f));
  }
  SECTION("AABB compare SA equality") {
    AABB aabb2 {
      vec3(1.1f),
      vec3(0.1f),
    };

    REQUIRE(aabb1.get_surface_area() == Approx(aabb2.get_surface_area()));
  }
  SECTION("AABB compare SA greater") {
    AABB aabb2 {
      vec3(1.0f),
      vec3(0.1f),
    };

    REQUIRE(aabb1.get_surface_area() > aabb2.get_surface_area());
  }
}

TEST_CASE("AABB SAH", "[aabb]") {
  AABB aabb1 {
    vec3(1.0f),
    vec3(0.0f),
  };

  SECTION("AABB calculate SAH") {
    REQUIRE(aabb1.get_cost(5) == Approx(30.0f));
  }
  SECTION("AABB compare SAH equality") {
    AABB aabb2 {
      vec3(1.1f),
      vec3(0.1f),
    };

    REQUIRE(aabb1.get_cost(5) == Approx(aabb2.get_cost(5)));
  }
  SECTION("AABB compare SAH greater") {
    AABB aabb2 {
      vec3(1.0f),
      vec3(0.1f),
    };

    REQUIRE(aabb1.get_cost(6) > aabb2.get_cost(5));
  }
}

TEST_CASE("AABB center", "[aabb]") {
  AABB aabb {
    vec3(1.0f),
    vec3(0.0f),
  };

  REQUIRE(aabb.get_center() == vec3(0.5f));
}

TEST_CASE("AABB grow", "[aabb]") {
  SECTION("AABB grow from smallest") {
    AABB aabb1 {
      vec3(-std::numeric_limits<float>::max()),
      vec3(std::numeric_limits<float>::max()),
    };
    AABB aabb2 {
      vec3(1.0f),
      vec3(0.0f),
    };
    aabb1.grow(aabb2);
    REQUIRE(aabb1 == aabb2);
  }

  SECTION("AABB grow bottom") {
    AABB aabb1 {
      vec3(1.0f),
      vec3(0.0f),
    };
    AABB aabb2 {
      vec3(1.0f),
      vec3(-1.0f),
    };
    aabb1.grow(aabb2);
    REQUIRE(aabb1 == aabb2);
  }

  SECTION("AABB no grow") {
    AABB aabb1 {
      vec3(1.0f),
      vec3(-1.0f),
    };
    AABB aabb_before = aabb1;
    AABB aabb2 {
      vec3(0.9f),
      vec3(0.0f),
    };
    aabb1.grow(aabb2);
    REQUIRE(aabb1 == aabb_before);
  }

  SECTION("AABB grow certain axes") {
    AABB aabb1 {
      vec3(1.0f),
      vec3(-1.0f),
    };
    AABB aabb2 {
      vec3(0.2f, 3.4f, 1.1f),
      vec3(0.0f, -2.0f, 1.0f),
    };
    AABB aabb_expected {
      vec3(1.0f, 3.4f, 1.1f),
      vec3(-1.0f, -2.0f, -1.0f),
    };
    aabb1.grow(aabb2);
    REQUIRE(aabb1 == aabb_expected);
  }
}

TEST_CASE("AABB shrink", "[aabb]") {
  SECTION("AABB shrink from largest") {
    AABB aabb1 {
      vec3(std::numeric_limits<float>::max()),
      vec3(-std::numeric_limits<float>::max()),
    };
    AABB aabb2 {
      vec3(1.0f),
      vec3(0.0f),
    };
    aabb1.shrink(aabb2);
    REQUIRE(aabb1 == aabb2);
  }

  SECTION("AABB shrink bottom") {
    AABB aabb1 {
      vec3(1.0f),
      vec3(-1.0f),
    };
    AABB aabb2 {
      vec3(1.0f),
      vec3(0.0f),
    };
    aabb1.shrink(aabb2);
    REQUIRE(aabb1 == aabb2);
  }

  SECTION("AABB no shrink") {
    AABB aabb1 {
      vec3(0.9f),
      vec3(0.0f),
    };
    AABB aabb_before = aabb1;
    AABB aabb2 {
      vec3(1.0f),
      vec3(-1.0f),
    };
    
    aabb1.shrink(aabb2);
    REQUIRE(aabb1 == aabb_before);
  }

  SECTION("AABB shrink certain axes") {
    AABB aabb1 {
      vec3(1.0f),
      vec3(-1.0f),
    };
    AABB aabb2 {
      vec3(0.2f, 3.4f, 1.1f),
      vec3(0.0f, -2.0f, 0.9f),
    };
    AABB aabb_expected {
      vec3(0.2f, 1.0f, 1.0f),
      vec3(0.0f, -1.0f, 0.9f),
    };
    aabb1.shrink(aabb2);
    REQUIRE(aabb1 == aabb_expected);
  }
}