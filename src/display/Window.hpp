/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
#include "glad/gl.h"

#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

#include <utility>

class WindowBase
{
public:
    auto ShouldClose() const noexcept { return glfwWindowShouldClose(window_); }
    glm::ivec2 GetWindowWidthAndHeight() const noexcept
    {
        int width, height;
        glfwGetWindowSize(window_, &width, &height);
        return { width, height };
    }
    glm::ivec2 GetWidthAndHeight() const noexcept
    {
        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);
        return { width, height };
    }
    glm::ivec2 GetPosition() const noexcept
    {
        int x, y;
        glfwGetWindowPos(window_, &x, &y);
        return { x, y };
    }
    auto SetPosition(glm::ivec2 coord) const noexcept
    {
        glfwSetWindowPos(window_, coord.x, coord.y);
    }
    // This could be better if we use Template Method Pattern to organize. But
    // to maximize performance (since this method is normally called every
    // frame), we let inherited class to call this manually and disable vptr.
    void BeginNewFrame() const noexcept;
    void EndCurrentFrame() const noexcept;
    auto GetRawHandle() const noexcept { return window_; }
    void Resize(glm::ivec2 size) { glfwSetWindowSize(window_, size.x, size.y); }

protected:
    static void SetBasicHints();
    void LoadContext();

    WindowBase() = default;
    ~WindowBase();

    GLFWwindow *window_ = nullptr;
};

template<typename T>
class WindowGuard
{
    const T &window_;

public:
    WindowGuard(const T &window) : window_{ window }
    {
        window_.BeginNewFrame();
    }

    ~WindowGuard() { window_.EndCurrentFrame(); }
};

// In MainWindow, we disable imgui draw.
class MainWindow : public WindowBase
{
public:
    static MainWindow &GetInstance();
    static void Wakeup() { glfwPostEmptyEvent(); }
    static void WaitForEvents() { glfwWaitEvents(); };
    static void NewEvents() { glfwPollEvents(); }

private:
    MainWindow();
    ~MainWindow();
};

class EventGuard
{
public:
    EventGuard() = default;
    ~EventGuard() { MainWindow::NewEvents(); }
};

class WindowForImGui : public WindowBase
{
public:
    WindowForImGui(int width, int height);
    ~WindowForImGui();

    void BeginNewFrame() const noexcept;
    void EndCurrentFrame() const noexcept;
};