#include <iostream>

#include "core/raytracer.h"

int main() {
  try {
    Raytracer rt;
    rt.raytrace();
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
