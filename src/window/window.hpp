#include "scene/scene.hpp"

struct GLFWwindow;

namespace nova {

class Window {
public:
  Window(bool headless = false);
  ~Window();

  void main_loop();

private:
  void initialize_scene();
  void display_menu();
  void display_scene_settings();
  void display_render();
  void handle_keyboard();
  void handle_mouse_drag();
  void handle_mouse_wheel();
  bool in_render_box();

  bool headless;
  GLFWwindow* window;
  Scene scene;
  int width;
  int height;
  float menu_height;
};

}