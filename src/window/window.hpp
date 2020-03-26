struct GLFWwindow;

class Window {
public:
  Window();
  ~Window();

  void main_loop();

private:
  void display_menu();
  void display_details_settings();
  void display_render();

  GLFWwindow* window;
  int width;
  int height;
  float menu_height;
};