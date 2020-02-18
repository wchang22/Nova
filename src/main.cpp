#include <iostream>

#include "core/raytracer.h"
#include "backend/types.h"

int main() {
  try {
    Raytracer rt(1280, 720, "raytrace");
    rt.raytrace();
  } catch(const Error& e) {
    std::cerr << e.what() << ": " << get_error_string(e.err()) << std::endl;
    return 1;
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
