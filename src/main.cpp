#include <iostream>

#include "backend/types.hpp"
#include "window/window.hpp"

int main() {
  try {
    Window window;
    window.main_loop();
  } catch (const Error& e) {
    std::cerr << e.what() << ": " << get_error_string(e.err()) << std::endl;
    return 1;
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
