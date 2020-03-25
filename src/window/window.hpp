struct GLFWwindow;

class Window {
public:
  Window();
  ~Window();

  void main_loop();

private:
  GLFWwindow* window;
  int width;
  int height;
};