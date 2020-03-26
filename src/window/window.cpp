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

#include "util/exception/exception.hpp"

const ImVec4 BG_COLOR(91.0f / 255.0f, 87.0f / 255.0f, 142.0f / 255.0f, 1.0f);
const ImVec4 HEADER_COLOR(53.0f / 255.0f, 53.0f / 255.0f, 70.0f / 255.0f, 1.0f);
const ImVec4 BUTTON_COLOR(49.0f / 255.0f, 49.0f / 255.0f, 104.0f / 255.0f, 1.0f);
const ImVec4 INPUT_COLOR(31.0f / 255.0f, 31.0f / 255.0f, 31.0f / 255.0f, 1.0f);
const ImVec4 ERROR_COLOR(1.0f, 0.0f, 0.0f, 1.0f);
const ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_NoCollapse |
                                      ImGuiWindowFlags_NoBringToFrontOnFocus;

Window::Window() {
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

  width = glfwGetVideoMode(glfwGetPrimaryMonitor())->width;
  height = glfwGetVideoMode(glfwGetPrimaryMonitor())->height;
  window = glfwCreateWindow(width, height, "Nova", nullptr, nullptr);

  if (!window) {
    glfwTerminate();
    throw WindowException("Failed to create GLFW Window");
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  // Initialize OpenGL loader
  if (gladLoadGL() == 0) {
    throw WindowException("Failed to initialize OpenGL loader");
  }

  // Setup Dear ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // Default styles
  ImGui::StyleColorsClassic();
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1);
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
  scene.cleanup_texture();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
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
  float window_width = 0.25f * width;
  float window_height = 1.0f * height - menu_height;
  auto& io = ImGui::GetIO();
  auto& style = ImGui::GetStyle();

  static std::string model_path = scene.get_model_path();
  static bool model_path_error = false;
  static std::array<float, 3> camera_position = scene.get_camera_position();
  static std::array<float, 3> camera_forward = scene.get_camera_forward();
  static std::array<float, 3> camera_up = scene.get_camera_up();
  static float camera_fovy = scene.get_camera_fovy();
  static std::array<float, 3> light_position = scene.get_light_position();
  static std::array<float, 3> light_intensity = scene.get_light_intensity();
  static int ray_bounces = scene.get_ray_bounces();
  static std::array<float, 3> shading_diffuse = scene.get_shading_diffuse();
  static float shading_metallic = scene.get_shading_metallic();
  static float shading_roughness = scene.get_shading_roughness();
  static float shading_ambient_occlusion = scene.get_shading_ambient_occlusion();

  if (ImGui::Begin("Details and Settings", nullptr, WINDOW_FLAGS)) {
    ImGui::SetWindowPos({ 0.0f, menu_height }, true);
    ImGui::SetWindowSize({ window_width, window_height }, true);

    ImGui::TextWrapped("%.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

    if (ImGui::CollapsingHeader("Model##SceneSettings", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (model_path_error) {
        ImGui::PushStyleColor(ImGuiCol_Border, ERROR_COLOR);
      }
      ImGui::InputText("Path##Model", &model_path);
      if (model_path_error) {
        ImGui::PopStyleColor();
      }
      model_path = scene.set_model_path(model_path);
    }

    if (ImGui::CollapsingHeader("Camera##SceneSettings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::InputFloat3("Position##Camera", camera_position.data());
      ImGui::InputFloat3("Forward##Camera", camera_forward.data());
      ImGui::InputFloat3("Up##Camera", camera_up.data());
      ImGui::InputFloat("FOVy##Camera", &camera_fovy);

      camera_position = scene.set_camera_position(camera_position);
      camera_forward = scene.set_camera_forward(camera_forward);
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

    ImGui::Indent(window_width / 2.0f - 2.0f * style.FramePadding.x);
    if (ImGui::Button("Update##SceneSettings", { window_width / 2.0f, 0 })) {
      scene.set_width(width);
      scene.set_height(height);
      try {
        scene.render();
        model_path_error = false;
      } catch (const ModelException& e) {
        model_path_error = true;
      }
    }

    ImGui::End();
  }
}

void Window::display_render() {
  float window_width = 0.75f * width;
  float window_height = 1.0f * height - menu_height;
  float image_width = window_width;
  float image_height = window_width * height / width;
  ImVec2 image_size { window_width, window_width * height / width };
  ImVec2 image_margin { (window_width - image_width) / 2.0f,
                        (window_height - image_height) / 2.0f };

  ImGui::Begin("Render", nullptr, WINDOW_FLAGS);

  ImGui::SetWindowPos({ width - window_width, menu_height }, true);
  ImGui::SetWindowSize({ window_width, window_height }, true);

  ImGui::SetCursorPos(image_margin);
  ImGui::Image(reinterpret_cast<ImTextureID>(scene.get_scene_texture_id()), image_size);

  ImGui::End();
}

void Window::main_loop() {
  while (!glfwWindowShouldClose(window)) {
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