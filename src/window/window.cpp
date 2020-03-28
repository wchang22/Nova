#include "window.hpp"

#include <iostream>

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <imgui/imgui.h>
#include <imgui/imgui_stdlib.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/ImGuiFileDialog.h>

#include "constants.hpp"
#include "util/exception/exception.hpp"
#include "util/profiling/profiling.hpp"

const ImVec4 BG_COLOR(0.2f, 0.2f, 0.2f, 1.0f);
const ImVec4 HEADER_COLOR(53.0f / 255.0f, 53.0f / 255.0f, 70.0f / 255.0f, 1.0f);
const ImVec4 BUTTON_COLOR(49.0f / 255.0f, 49.0f / 255.0f, 104.0f / 255.0f, 1.0f);
const ImVec4 INPUT_COLOR(31.0f / 255.0f, 31.0f / 255.0f, 31.0f / 255.0f, 1.0f);
const ImVec4 ERROR_COLOR(1.0f, 0.0f, 0.0f, 1.0f);
constexpr float LEFT_PANEL_PERCENTAGE = 0.25f;
constexpr float RIGHT_PANEL_PERCENTAGE = 1.0f - LEFT_PANEL_PERCENTAGE;
constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                          ImGuiWindowFlags_NoCollapse |
                                          ImGuiWindowFlags_NoBringToFrontOnFocus;

Window::Window(bool headless) : headless(headless) {
  // Don't setup anything if no window needed
  if (headless) {
    return;
  }

  // Setup GLFW and window
  glfwSetErrorCallback([](int error, const char* description) {
    std::cerr << "GLFW Error: " << error << ": " << description << std::endl;
  });
  if (!glfwInit()) {
    throw WindowException("GLFW failed to initialize");
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  const GLFWvidmode* vid_mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
  width = vid_mode->width;
  height = vid_mode->height;
  window = glfwCreateWindow(width, height, "Nova", nullptr, nullptr);

  if (!window) {
    glfwTerminate();
    throw WindowException("Failed to create GLFW Window");
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  // Initialize OpenGL loader
  if (!gladLoadGL()) {
    throw WindowException("Failed to initialize OpenGL loader");
  }

  // Setup Dear ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.Fonts->AddFontFromFileTTF(FONT_PATH, FONT_SIZE);

  // Default styles
  ImGui::StyleColorsClassic();
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, BG_COLOR);
  ImGui::PushStyleColor(ImGuiCol_Header, HEADER_COLOR);
  ImGui::PushStyleColor(ImGuiCol_Button, BUTTON_COLOR);
  ImGui::PushStyleColor(ImGuiCol_FrameBg, INPUT_COLOR);

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  scene.init_texture();
}

Window::~Window() {
  if (!headless) {
    scene.cleanup_texture();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
  }
}

void display_input_text_error(bool error_flag, const char* label, std::string& str) {
  // Display border highlight if error
  if (error_flag) {
    ImGui::PushStyleColor(ImGuiCol_Border, ERROR_COLOR);
  }
  ImGui::InputText(label, &str);
  if (error_flag) {
    ImGui::PopStyleColor();
  }
}

void display_file_dialog(float button_indent,
                         float button_width,
                         const char* button_label,
                         const char* filters,
                         std::string& path) {
  static ImGuiFileDialog* dialog_inst = ImGuiFileDialog::Instance();
  std::string dialog_key = button_label + std::string("Key");

  // Display button and file dialog popup on button click
  ImGui::Indent(button_indent);
  if (ImGui::Button(button_label, { button_width, 0.0f })) {
    dialog_inst->OpenDialog(dialog_key, "Browse", filters, ".");
  }
  ImGui::Indent(-button_indent);
  if (dialog_inst->FileDialog(dialog_key)) {
    if (dialog_inst->IsOk) {
      path = dialog_inst->GetFilepathName();
    }
    dialog_inst->CloseDialog(dialog_key);
  }
}

void Window::display_menu() {
  if (ImGui::BeginMainMenuBar()) {
    menu_height = ImGui::GetWindowHeight();
    if (ImGui::BeginMenu("File##Menu")) {
      if (ImGui::MenuItem("Quit##Menu")) {
        glfwSetWindowShouldClose(window, true);
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

void Window::display_scene_settings() {
  float window_width = LEFT_PANEL_PERCENTAGE * width;
  float window_height = height - menu_height;
  ImGuiIO& io = ImGui::GetIO();
  ImGuiStyle& style = ImGui::GetStyle();

  const float button_width =
    window_width * 0.5f - 1.5f * style.FramePadding.x - 4.0f * style.FrameBorderSize;
  const float button_indent =
    window_width - 2.0f * style.FramePadding.x - 4.0f * style.FrameBorderSize - button_width;

  // UI element variables
  static bool real_time = false;
  static std::array<int, 2> output_dimensions = scene.get_output_dimensions();
  static std::string output_file_path = scene.get_output_file_path();
  static bool file_path_error = false;
  static std::string model_path = scene.get_model_path();
  static bool model_path_error = false;
  static std::array<float, 3> camera_position = scene.get_camera_position();
  static std::array<float, 3> camera_target = scene.get_camera_target();
  static std::array<float, 3> camera_up = scene.get_camera_up();
  static float camera_fovy = scene.get_camera_fovy();
  static std::array<float, 3> light_position = scene.get_light_position();
  static std::array<float, 3> light_intensity = scene.get_light_intensity();
  static int ray_bounces = scene.get_ray_bounces();
  static std::array<float, 3> shading_diffuse = scene.get_shading_diffuse();
  static float shading_metallic = scene.get_shading_metallic();
  static float shading_roughness = scene.get_shading_roughness();
  static float shading_ambient_occlusion = scene.get_shading_ambient_occlusion();

  // Render lambdas that check for errors
  const auto render_to_screen = [&]() {
    if (model_path_error) {
      return;
    }
    try {
      scene.render_to_screen();
      model_path_error = false;
    } catch (const ModelException& e) {
      model_path_error = true;
    }
  };
  const auto render_to_image = [&]() {
    if (model_path_error || file_path_error) {
      return;
    }
    try {
      scene.render_to_image();
      model_path_error = false;
      file_path_error = false;
    } catch (const ModelException& e) {
      model_path_error = true;
    } catch (const ImageException& e) {
      file_path_error = true;
    }
  };

  if (ImGui::Begin("Details and Settings", nullptr, WINDOW_FLAGS)) {
    ImGui::SetWindowPos({ 0.0f, menu_height }, true);
    ImGui::SetWindowSize({ window_width, window_height }, true);

    ImGui::TextWrapped("%.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

    if (ImGui::CollapsingHeader("Rendering##SceneSettings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Checkbox("Enable Real-Time##Rendering", &real_time);
      ImGui::InputInt2("Resolution##Rendering", output_dimensions.data());

      display_input_text_error(file_path_error, "Save Path##Rendering", output_file_path);
      display_file_dialog(button_indent, button_width, "Browse##Rendering", IMAGE_EXTENSION,
                          output_file_path);

      output_dimensions = scene.set_output_dimensions(output_dimensions);
    }

    if (ImGui::CollapsingHeader("Model##SceneSettings", ImGuiTreeNodeFlags_DefaultOpen)) {
      display_input_text_error(model_path_error, "Path##Model", model_path);
      display_file_dialog(button_indent, button_width, "Browse##Model", MODEL_FILE_TYPES,
                          model_path);
    }

    if (ImGui::CollapsingHeader("Camera##SceneSettings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::InputFloat3("Position##Camera", camera_position.data());
      ImGui::InputFloat3("Target##Camera", camera_target.data());
      ImGui::InputFloat3("Up##Camera", camera_up.data());
      ImGui::InputFloat("FOV##Camera", &camera_fovy);

      camera_position = scene.set_camera_position(camera_position);
      camera_target = scene.set_camera_target(camera_target);
      camera_up = scene.set_camera_up(camera_up);
      camera_fovy = scene.set_camera_fovy(camera_fovy);
    }

    if (ImGui::CollapsingHeader("Light##SceneSettings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::InputFloat3("Position##Light", light_position.data());
      ImGui::InputFloat3("Intensity##Light", light_intensity.data());

      light_position = scene.set_light_position(light_position);
      light_intensity = scene.set_light_intensity(light_intensity);
    }

    if (ImGui::CollapsingHeader("Ray Bounces##SceneSettings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::InputInt("Number##RayBounces", &ray_bounces);

      ray_bounces = scene.set_ray_bounces(ray_bounces);
    }

    if (ImGui::CollapsingHeader("Shading Defaults##SceneSettings",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::InputFloat3("Diffuse##ShadingDefaults", shading_diffuse.data());
      ImGui::InputFloat("Metallic##ShadingDefaults", &shading_metallic);
      ImGui::InputFloat("Roughness##ShadingDefaults", &shading_roughness);
      ImGui::InputFloat("AO##ShadingDefaults", &shading_ambient_occlusion);

      shading_diffuse = scene.set_shading_diffuse(shading_diffuse);
      shading_metallic = scene.set_shading_metallic(shading_metallic);
      shading_roughness = scene.set_shading_roughness(shading_roughness);
      shading_ambient_occlusion = scene.set_shading_ambient_occlusion(shading_ambient_occlusion);
    }

    if (ImGui::Button("Save Image##SceneSettings", { button_width, 0.0f })) {
      output_file_path = scene.set_output_file_path(output_file_path);
      model_path = scene.set_model_path(model_path);
      model_path_error = false;
      file_path_error = false;
      render_to_image();
    }
    ImGui::SameLine();
    if (ImGui::Button("Update##SceneSettings", { button_width, 0.0f })) {
      model_path = scene.set_model_path(model_path);
      model_path_error = false;
      // No need to render on click if already rendering
      if (!real_time) {
        render_to_screen();
      }
    }

    ImGui::End();
  }

  // Handle IO events and render per frame if real time
  if (real_time) {
    handle_mouse_drag();
    handle_mouse_wheel();
    handle_keyboard();
    // Update camera position due to IO events
    camera_position = scene.get_camera_position();
    render_to_screen();
  }
}

void Window::display_render() {
  float window_width = RIGHT_PANEL_PERCENTAGE * width;
  float window_height = height - menu_height;
  ImVec2 image_size { window_width, window_width * height / width };
  ImVec2 image_margin { 0.0f, (window_height - image_size.y) * 0.5f };

  ImGui::Begin("Render", nullptr, WINDOW_FLAGS);

  ImGui::SetWindowPos({ width - window_width, menu_height }, true);
  ImGui::SetWindowSize({ window_width, window_height }, true);

  // Display render as an image
  ImGui::SetCursorPos(image_margin);
  ImGui::Image(reinterpret_cast<ImTextureID>(scene.get_scene_texture_id()), image_size);

  ImGui::End();
}

// Checks if mouse is in render image
bool Window::in_render_box() {
  float window_width = RIGHT_PANEL_PERCENTAGE * width;
  float window_height = height - menu_height;
  ImVec2 image_size { window_width, window_width * height / width };
  ImVec2 image_margin { width - window_width, (window_height - image_size.y) * 0.5f };

  ImVec2 mouse_pos = ImGui::GetMousePos();

  return mouse_pos.x <= image_margin.x + image_size.x && mouse_pos.x >= image_margin.x &&
         mouse_pos.y <= image_margin.y + image_size.y && mouse_pos.y >= image_margin.y;
}

void Window::handle_keyboard() {
  if (!in_render_box()) {
    return;
  }

  constexpr float MOUSE_SENSITIVITY = 100.0f;
  ImGuiIO& io = ImGui::GetIO();
  // Adjust camera speed according to frame rate
  float camera_speed = MOUSE_SENSITIVITY / io.Framerate;

  if (ImGui::IsKeyPressed(GLFW_KEY_W) || ImGui::IsKeyPressed(GLFW_KEY_UP)) {
    scene.move_camera(Camera::Direction::UP, camera_speed);
  }
  if (ImGui::IsKeyPressed(GLFW_KEY_S) || ImGui::IsKeyPressed(GLFW_KEY_DOWN)) {
    scene.move_camera(Camera::Direction::UP, -camera_speed);
  }
  if (ImGui::IsKeyPressed(GLFW_KEY_A) || ImGui::IsKeyPressed(GLFW_KEY_LEFT)) {
    scene.move_camera(Camera::Direction::RIGHT, -camera_speed);
  }
  if (ImGui::IsKeyPressed(GLFW_KEY_D) || ImGui::IsKeyPressed(GLFW_KEY_RIGHT)) {
    scene.move_camera(Camera::Direction::RIGHT, camera_speed);
  }
}

void Window::handle_mouse_drag() {
  if (!in_render_box()) {
    return;
  }

  constexpr float MOUSE_SENSITIVITY = 0.5f;
  ImVec2 delta = ImGui::GetMouseDragDelta();

  // TODO std::array, glm::vec conversions, typedefs
  scene.move_camera(Camera::Direction::RIGHT, -delta.x * MOUSE_SENSITIVITY);
  scene.move_camera(Camera::Direction::UP, delta.y * MOUSE_SENSITIVITY);
  ImGui::ResetMouseDragDelta();
}

void Window::handle_mouse_wheel() {
  if (!in_render_box()) {
    return;
  }

  constexpr float MOUSE_SENSITIVITY = 20.0f;
  auto& io = ImGui::GetIO();
  float scroll_amount = io.MouseWheel;
  if (scroll_amount == 0) {
    return;
  }
  // Adjust camera speed according to frame rate
  float camera_speed = MOUSE_SENSITIVITY / io.Framerate * scroll_amount;

  scene.move_camera(Camera::Direction::FORWARD, camera_speed);
}

void Window::main_loop() {
  if (headless) {
    PROFILE_SCOPE("Main Loop");
    scene.render_to_image();
  } else {
    while (!glfwWindowShouldClose(window)) {
      PROFILE_SCOPE("Main Loop");

      glfwPollEvents();

      // New frame
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // Displays
      display_menu();
      display_scene_settings();
      display_render();

      // Rendering
      ImGui::Render();
      glfwGetFramebufferSize(window, &width, &height);
      glViewport(0, 0, width, height);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(window);
    }
  }
}