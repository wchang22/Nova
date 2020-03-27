#include "scene/scene.hpp"

struct GLFWwindow;

class Window {
public:
  Window();
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

  GLFWwindow* window;
  Scene scene;
  int width;
  int height;
  float menu_height;
};