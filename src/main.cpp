#include <CLI11.hpp>
#include <iostream>

#include "backend/types.hpp"
#include "constants.hpp"
#include "window/window.hpp"

int main(int argc, char** argv) {
  CLI::App app(APP_DESCRIPTION);

  bool headless;

  app.add_flag("--headless", headless, "Launch Nova without a GUI");

  try {
    app.parse(argc, argv);
    nova::Window window(headless);
    window.main_loop();
  } catch (const CLI::Error& e) {
    app.exit(e);
  } catch (const nova::Error& e) {
    std::cerr << e.what() << ": " << nova::get_error_string(e.err()) << std::endl;
    return 1;
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
