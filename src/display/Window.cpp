/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "Window.hpp"
#include "Utils.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef _WIN32

#include <IntSafe.h>

extern "C"
{
    _declspec(dllexport) DWORD NvOptimusEnablement = 1;
}

#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"

#endif

#include <format>
#include <iostream>

#define NEED_GLFW_LOG_ERROR

// clang-format off
static void APIENTRY GLDebugMessageCallback(GLenum source, GLenum type, GLuint id,
                                            GLenum severity, GLsizei length,
                                            const GLchar *message, const void *param)
{
	std::string_view source_, type_, severity_, message_{ message, (std::size_t)length };

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             source_ = "API";             break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   source_ = "WINDOW_SYSTEM";   break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: source_ = "SHADER_COMPILER"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     source_ = "THIRD_PARTY";     break;
	case GL_DEBUG_SOURCE_APPLICATION:     source_ = "APPLICATION";     break;
	case GL_DEBUG_SOURCE_OTHER:           source_ = "OTHER";           break;
	default:                              source_ = "<SOURCE>";        break;
	}

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               type_ = "ERROR";               break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: type_ = "DEPRECATED_BEHAVIOR"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  type_ = "UNDEFINED_BEHAVIOR";  break;
	case GL_DEBUG_TYPE_PORTABILITY:         type_ = "PORTABILITY";         break;
	case GL_DEBUG_TYPE_PERFORMANCE:         type_ = "PERFORMANCE";         break;
	case GL_DEBUG_TYPE_OTHER:               type_ = "OTHER";               break;
	case GL_DEBUG_TYPE_MARKER:              type_ = "MARKER";              break;
	default:                                type_ = "<TYPE>";              break;
	}

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         severity_ = "HIGH";         break;
	case GL_DEBUG_SEVERITY_MEDIUM:       severity_ = "MEDIUM";       break;
	case GL_DEBUG_SEVERITY_LOW:          severity_ = "LOW";          break;
    // Notification isn't printed.
	case GL_DEBUG_SEVERITY_NOTIFICATION: severity_ = "NOTIFICATION"; return;
	default:                             severity_ = "<SEVERITY>";   break;
	}

    // TODO: Can we use GLGetError here?
    std::cerr << std::format("{}: GL {} {} ({}): {}.\n", id, severity_, type_, 
                 source_, message_);
}
// clang-format on

[[maybe_unused]] static void LogGLFWError(int errorCode,
                                          const char *description)
{
    std::cout << std::format("GLFW error[code {}]: {}\n", errorCode,
                             description);
}

void WindowBase::SetBasicHints()
{
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void WindowBase::BeginNewFrame() const noexcept
{
    glfwMakeContextCurrent(window_);
}

void WindowBase::EndCurrentFrame() const noexcept
{
    glfwSwapBuffers(window_);
}

WindowBase::~WindowBase()
{
    if (window_)
        glfwDestroyWindow(window_);
}

void WindowBase::LoadContext()
{
    CheckError(window_, "Unable to create GLFW window");
    glfwMakeContextCurrent(window_);
    CheckError(gladLoadGL(glfwGetProcAddress), "Unable to load GL by glad");
    glfwSwapInterval(0);
}

MainWindow::MainWindow()
{
    SetBasicHints();
#ifdef NEED_GLFW_LOG_ERROR
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    glfwSetErrorCallback(LogGLFWError);
#endif

    window_ = glfwCreateWindow(1024, 1024, "Test", nullptr, nullptr);
    LoadContext();

    // glGetString returns GLubyte strangely, which is normally unsigned char
    // and is probably not printable directly; we convert it to const char*.
    std::cerr << std::format("Vendor: {}, Renderer: {}\n",
                             (const char *)glGetString(GL_VENDOR),
                             (const char *)glGetString(GL_RENDERER));

#ifdef NEED_GLFW_LOG_ERROR
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(GLDebugMessageCallback, NULL);
#endif
}

MainWindow::~MainWindow()
{ // Since it's illegal to destroy after termiation.
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
}

MainWindow &MainWindow::GetInstance()
{
    CheckError(glfwInit(), "Fail to initialize GLFW.");
#ifdef NEED_GLFW_LOG_ERROR
    glfwSetErrorCallback(LogGLFWError);
#endif

    static MainWindow instance{};
    return instance;
}

// ------------------- WindowForImGui -------------------

WindowForImGui::WindowForImGui(int width, int height)
{
    SetBasicHints();
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    window_ = glfwCreateWindow(width, height, "", nullptr, nullptr);
    LoadContext();

#ifdef _WIN32
    glfwHideWindow(window_);
    SetWindowLong(glfwGetWin32Window(window_), GWL_EXSTYLE, WS_EX_TOOLWINDOW);
    glfwShowWindow(window_);
#endif

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    // Don't save configs for window.
    io.IniFilename = nullptr;
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 430");
}

WindowForImGui::~WindowForImGui()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void WindowForImGui::BeginNewFrame() const noexcept
{
    WindowBase::BeginNewFrame();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void WindowForImGui::EndCurrentFrame() const noexcept
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    WindowBase::EndCurrentFrame();
    // ImGui::UpdatePlatformWindows();
    // ImGui::RenderPlatformWindowsDefault();
    // glfwMakeContextCurrent(window_);
}