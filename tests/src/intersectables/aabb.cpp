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

TEST_CASE("AABB axis intersect", "[aabb]") {
  AABB aabb1 {
    vec3(-0.1f, 0.2f, 0.2f),
    vec3(-1.0f),
  };
  AABB aabb2 {
    vec3(1.0f),
    vec3(0.0f),
  };

  SECTION("AABB intersect right") {
    REQUIRE(!aabb1.intersects(aabb2, 0));
    REQUIRE(aabb1.intersects(aabb2, 1));
    REQUIRE(aabb1.intersects(aabb2, 2));
  }

  SECTION("AABB intersect left") {
    REQUIRE(!aabb2.intersects(aabb1, 0));
    REQUIRE(aabb2.intersects(aabb1, 1));
    REQUIRE(aabb2.intersects(aabb1, 2));
  }
}

TEST_CASE("AABB intersect", "[aabb]") {
  AABB aabb1 {
    vec3(-0.1f, 0.2f, 0.2f),
    vec3(-1.0f),
  };
  AABB aabb2 {
    vec3(1.0f),
    vec3(0.0f),
  };
  AABB aabb3 {
    vec3(0.2f),
    vec3(-1.0f),
  };

  SECTION("AABB intersect right") {
    REQUIRE(!aabb1.intersects(aabb2));
    REQUIRE(aabb2.intersects(aabb3));
  }

  SECTION("AABB intersect left") {
    REQUIRE(!aabb2.intersects(aabb1));
    REQUIRE(aabb3.intersects(aabb2));
  }
}

TEST_CASE("AABB is in", "[aabb]") {
  AABB aabb1 {
    vec3(0.2f),
    vec3(-1.0f),
  };
  AABB aabb2 {
    vec3(1.0f),
    vec3(0.0f),
  };
  AABB aabb3 {
    vec3(0.75f),
    vec3(0.25f),
  };

  SECTION("AABB is in") {
    REQUIRE(aabb3.is_in(aabb2));
    REQUIRE(aabb3.is_in(aabb3));
  }

  SECTION("AABB is not in") {
    REQUIRE(!aabb1.is_in(aabb2));
    REQUIRE(!aabb2.is_in(aabb1));
    REQUIRE(!aabb2.is_in(aabb3));
  }
}

TEST_CASE("AABB get intersection", "[aabb]") {
  AABB aabb1 {
    vec3(0.2f),
    vec3(-1.0f),
  };
  AABB aabb2 {
    vec3(1.0f),
    vec3(0.0f),
  };
  AABB intersection {
    vec3(0.2f),
    vec3(0.0f),
  };

  REQUIRE(aabb1.get_intersection(aabb2) == intersection);
  REQUIRE(aabb2.get_intersection(aabb1) == intersection);
}

TEST_CASE("AABB get union", "[aabb]") {
  AABB aabb1 {
    vec3(0.2f),
    vec3(-1.0f),
  };
  AABB aabb2 {
    vec3(1.0f),
    vec3(0.0f),
  };
  AABB aabb_union {
    vec3(1.0f),
    vec3(-1.0f),
  };

  REQUIRE(aabb1.get_union(aabb2) == aabb_union);
  REQUIRE(aabb2.get_union(aabb1) == aabb_union);
}

TEST_CASE("AABB chopped union", "[aabb]") {
  AABB aabb {
    vec3(192.92f),
    vec3(0.0f),
  };

  size_t num_chops = 13;
  float chop_size = 192.92f / num_chops;
  std::vector<std::vector<AABB>> chopped_aabbs(3, std::vector<AABB>(num_chops));

  for (int axis = 0; axis < 3; axis++) {
    for (size_t i = 0; i < num_chops; i++) {
      AABB chop = aabb;
      chop.bottom[axis] = i * chop_size;
      chop.top[axis] = (i + 1) * chop_size;
      chopped_aabbs[axis][i] = std::move(chop);
    }
  }

  SECTION("Chopped whole") {
    for (int axis = 0; axis < 3; axis++) {
      AABB chopped_union {
        vec3(-std::numeric_limits<float>::max()),
        vec3(std::numeric_limits<float>::max()),
      };

      for (size_t i = 0; i < num_chops; i++) {
        chopped_union.grow(chopped_aabbs[axis][i]);
      }

      REQUIRE(chopped_union == aabb);
    }
  }

  SECTION("Chopped left right") {
    for (int axis = 0; axis < 3; axis++) {
      AABB chopped_left {
        vec3(-std::numeric_limits<float>::max()),
        vec3(std::numeric_limits<float>::max()),
      };
      AABB chopped_right = chopped_left;

      for (size_t i = 0; i < num_chops / 2; i++) {
        chopped_left.grow(chopped_aabbs[axis][i]);
      }
      for (size_t i = num_chops / 2; i < num_chops; i++) {
        chopped_right.grow(chopped_aabbs[axis][i]);
      }

      float split = (num_chops / 2) * chop_size;
      AABB left = aabb;
      AABB right = aabb;
      left.top[axis] = split;
      right.bottom[axis] = split;

      REQUIRE(left == chopped_left);
      REQUIRE(right == chopped_right);
    }
  }
}