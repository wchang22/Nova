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

  GLFWwindow* window;
  Scene scene;
  int width;
  int height;
  float menu_height;
};