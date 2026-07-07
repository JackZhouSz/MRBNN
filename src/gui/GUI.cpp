/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "GUI.hpp"

#include "ConfigLoader.hpp"
#include "display/FileDialog.hpp"

#include "imgui.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"
#undef min
#undef max

#endif

namespace
{

namespace Detail
{
template<auto ImGuiBegin, auto ImGuiEnd>
class ImGuiScope
{
public:
    ImGuiScope(auto... args) : success_{ ImGuiBegin(args...) } {}
    bool IsSuccess() const noexcept { return success_; }
    explicit operator bool() { return success_; }
    // Other special members are disabled automatically.
    ~ImGuiScope()
    {
        if (success_)
            ImGuiEnd();
    }

private:
    bool success_;
};

} // namespace Detail

using ImGuiWindow = Detail::ImGuiScope<ImGui::Begin, ImGui::End>;
#define UI_SPLIT_POINT 125

template<typename T, typename... Args>
bool LeftLableUI(T UI, std::string_view labelImpl, Args... args)
{
    std::string label{ "##" };
    label.append(labelImpl);

    ImGui::Text(labelImpl.data());
    ImGui::SameLine(UI_SPLIT_POINT);
    return UI(label.c_str(), args...);
}

#define FastLeftLabelUI(UI, LABEL, ...)   \
    [&]() {                               \
        const char *label = "##" LABEL;   \
        ImGui::AlignTextToFramePadding(); \
        ImGui::Text(label + 2);           \
        ImGui::SameLine(UI_SPLIT_POINT);  \
        return UI(label, __VA_ARGS__);    \
    }()

template<typename T, int N>
class UBSanityProxy
{
    glm::vec<N, T> &vec_;
    T arr_[N];

public:
    UBSanityProxy(glm::vec<N, T> &vec) : vec_{ vec }
    {
        for (int i = 0; i < N; i++)
            arr_[i] = vec_[i];
    }

    operator T *() noexcept { return arr_; }

    ~UBSanityProxy()
    {
        for (int i = 0; i < N; i++)
            vec_[i] = arr_[i];
    }
};

template<typename T, typename U>
    requires std::is_enum_v<T> && std::is_integral_v<U>
class EnumUBSanityProxy
{
    T &enum_;
    U val_;

public:
    EnumUBSanityProxy(T &init_enum)
        : enum_{ init_enum }, val_{ static_cast<U>(init_enum) }
    { // Check whether overflow.
        assert(static_cast<T>(val_) == enum_);
    }

    operator U *() noexcept { return &val_; }

    ~EnumUBSanityProxy() { enum_ = static_cast<T>(val_); }
};

template<typename U, typename T>
auto MakeEnumUBSanityProxy(T &e)
{ // For tedious CTAD disallowing partial deduction.
    return EnumUBSanityProxy<T, U>{ e };
}

// (theta, phi, r)
glm::vec3 CartesianToSpherical(glm::vec3 coord)
{
    float r = glm::length(coord);
    float theta = std::acos(coord.y / r);
    float phi = std::atan2(coord.z, coord.x);

    return { theta, phi, r };
}

glm::vec3 SphericalToCartesian(glm::vec3 coord)
{
    float theta = coord.x, phi = coord.y, radius = coord.z;
    float x = radius * std::sin(theta) * std::cos(phi);
    float y = radius * std::cos(theta);
    float z = radius * std::sin(theta) * std::sin(phi);
    return glm::vec3(x, y, z);
}

} // namespace

void GUI::SyncCache()
{
    cacheData_.lightSphCoord = CartesianToSpherical(params_.lightDir);
    cacheData_.cameraSphCoord = CartesianToSpherical(params_.cameraPos);
}

glm::vec2 GUI::Draw()
{
    [[maybe_unused]] auto &io = ImGui::GetIO();
    bool resetFrameNum = false;

    ImGuiWindow window{ "Settings", nullptr,
                        ImGuiWindowFlags_NoMove |
                            ImGuiWindowFlags_NoDecoration |
                            ImGuiWindowFlags_NoResize };

    ImGui::SetWindowPos(ImVec2(0, 0));
    ImGui::SetWindowSize(ImVec2(375, 700));

    ImGui::Text("Resolution: [%d, %d]", cacheData_.fbWidth,
                cacheData_.fbHeight);
    ImGui::Text("FPS: %f", cacheData_.fps);
    ImGui::Text("Frame No.%d", params_.frameNum++);

    params_.cameraPos = SphericalToCartesian(cacheData_.cameraSphCoord);
    ImGui::Text("Camera position: [%.3f, %.3f, %.3f]", params_.cameraPos.x,
                params_.cameraPos.y, params_.cameraPos.z);
    ImGui::Text("Light position: [%.3f, %.3f, %.3f]", params_.lightDir.x,
                params_.lightDir.y, params_.lightDir.z);

    FastLeftLabelUI(ImGui::Combo, "Tone Mapping",
                    MakeEnumUBSanityProxy<int>(params_.toneMapping),
                    Description::s_toneMapping,
                    std::size(Description::s_toneMapping));
    resetFrameNum |= FastLeftLabelUI(
        ImGui::Combo, "Denoise", MakeEnumUBSanityProxy<int>(params_.denoise),
        Description::s_denoise, std::size(Description::s_denoise));
    ImGui::Separator();

    resetFrameNum |= FastLeftLabelUI(ImGui::SliderFloat, "HG param ",
                                     &params_.g, 0.0, 0.86f);
    // Well, practically we can use (float*)&vec by UB type punning.
    resetFrameNum |= FastLeftLabelUI(ImGui::SliderFloat3, "Albedo ",
                                     UBSanityProxy{ params_.albedo }, 0.0, 1.0);

    resetFrameNum |= FastLeftLabelUI(ImGui::SliderFloat, "Light altitude ",
                                     &cacheData_.lightSphCoord.x, 0,
                                     std::numbers::pi_v<float>);
    resetFrameNum |= FastLeftLabelUI(
        ImGui::SliderFloat, "Light azumith ", &cacheData_.lightSphCoord.y,
        -std::numbers::pi_v<float>, std::numbers::pi_v<float>);
    params_.lightDir =
        SphericalToCartesian(glm::vec3{ cacheData_.lightSphCoord, 1.0f });

    ImGui::Text("Light Color");
    resetFrameNum |= ImGui::ColorPicker3(
        "##Light Color", UBSanityProxy{ params_.lightColor },
        ImGuiColorEditFlags_::ImGuiColorEditFlags_Float |
            ImGuiColorEditFlags_::ImGuiColorEditFlags_NoAlpha |
            ImGuiColorEditFlags_::ImGuiColorEditFlags_HDR);

    ImGui::Separator();
    ImGui::Text("Volume Information");

    params_.recreateDensityTexture = FastLeftLabelUI(
        ImGui::Combo, "Interpolation",
        MakeEnumUBSanityProxy<int>(params_.densityInterpolation),
        Description::s_interpolation, std::size(Description::s_interpolation));
    resetFrameNum |= params_.recreateDensityTexture;

    bool refreshTrMap = false;
    refreshTrMap |= FastLeftLabelUI(ImGui::SliderInt, "Sample level ",
                                    &params_.trMapSampleLevel, 0, 8);
    refreshTrMap |= FastLeftLabelUI(ImGui::SliderInt, "Marching steps ",
                                    &params_.trMapMarchingSteps, 0, 128);
    params_.refreshTrMap = refreshTrMap;

    resetFrameNum |= refreshTrMap;
    resetFrameNum |= FastLeftLabelUI(ImGui::Checkbox, "Fast Direct Illum",
                                     &params_.fastDirectIllum);

    ImGui::Separator();
    bool skipSkyboxReset = false;
    if (ImGui::Button("Select working directory"))
    {
        if (auto path = OpenFolderDialog())
        {
            if (auto bakingDir = ConfiguationLoader::TryLoadSkybox(
                    *path / "skybox_init.json", params_);
                !bakingDir.empty())
            {
                cacheData_.skyboxBakingWorkDir = std::move(bakingDir);
            }

            cacheData_.workDir = std::move(path);
            resetFrameNum = true, skipSkyboxReset = true;
        }
    }
    else
    {
        cacheData_.workDir.reset();
    }

    if (ImGui::Button("Load menu configuration"))
    {
        if (auto path = OpenFileDialog())
        {
            LoadConfig(*path);
            resetFrameNum = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save menu configuration"))
    {
        if (auto path = OpenFileDialog())
            SaveConfig(*path);
    }

    ImGui::Separator();
    if (ImGui::Button("Select skybox HDRI"))
    {
        if (auto path = OpenFileDialog())
        {
            params_.skybox = EnvironmentMap{ *path };
            resetFrameNum = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Select baking"))
    {
        if (auto path = OpenFolderDialog())
        {
            cacheData_.skyboxBakingWorkDir = std::move(path);
            resetFrameNum = true;
        }
    }
    else if (!skipSkyboxReset)
    {
        cacheData_.skyboxBakingWorkDir.reset();
    }

    resetFrameNum |= FastLeftLabelUI(ImGui::Checkbox, "Enable baking",
                                     &params_.enableSkyboxBaking);
    resetFrameNum |= FastLeftLabelUI(ImGui::Checkbox, "Stochastic illum",
                                     &params_.enableSkybox);
    resetFrameNum |= FastLeftLabelUI(ImGui::SliderFloat, "Exposure",
                                     &params_.skybox.Exposure(), 0.0, 5.0);

    ImGui::Separator();
    ImGui::Text("Backward Compatibility Options");
    resetFrameNum |= FastLeftLabelUI(ImGui::Checkbox, "Disable LightEnc",
                                     &params_.excludeLightEncoding);
    resetFrameNum |=
        FastLeftLabelUI(ImGui::Combo, "Direct Illum",
                        MakeEnumUBSanityProxy<int>(params_.compatibility),
                        Description::s_lightCompatibility,
                        std::size(Description::s_lightCompatibility));

    if (resetFrameNum)
        params_.frameNum = 0;

    auto finalSize = ImGui::GetWindowSize();
    return glm::vec2{ finalSize.x, finalSize.y };
}

static void AttachWindow(GLFWwindow *window, glm::ivec2 position, int width)
{
    glfwSetWindowPos(window, position.x + width, position.y);
}

static void GroupWindow(GLFWwindow *targetHandle, GLFWwindow *srcHandle)
{
#ifdef _WIN32
    HWND srcRawHandle = glfwGetWin32Window(srcHandle);
    HWND targetRawHandle = glfwGetWin32Window(targetHandle);
    SetWindowLongPtr(targetRawHandle, GWLP_HWNDPARENT, (LONG_PTR)srcRawHandle);
#endif
}

void GUI::RegisterCallbacks(const WindowBase &targetWindow,
                            const WindowBase &sourceWindow)
{
    auto targetHandle = targetWindow.GetRawHandle(),
         srcHandle = sourceWindow.GetRawHandle();
    GroupWindow(targetHandle, srcHandle);

    auto initSize = sourceWindow.GetWidthAndHeight();
    cacheData_.fbWidth = initSize.x, cacheData_.fbHeight = initSize.y;

    callbackData_.targetWindow = targetHandle;
    callbackData_.guiPtr = this;
    glfwSetWindowUserPointer(srcHandle, &(this->callbackData_));
    glfwSetWindowPosCallback(srcHandle, [](auto srcWindow, auto x, auto y) {
        auto ptr = (CallbackData *)glfwGetWindowUserPointer(srcWindow);
        int width;
        glfwGetWindowSize(srcWindow, &width, nullptr);
        AttachWindow(ptr->targetWindow, { x, y }, width);
    });
    glfwSetWindowSizeCallback(
        srcHandle, [](auto srcWindow, auto width, auto height) {
            auto ptr = (CallbackData *)glfwGetWindowUserPointer(srcWindow);
            int x, y;
            glfwGetWindowPos(srcWindow, &x, &y);
            AttachWindow(ptr->targetWindow, { x, y }, width);
        });

    glfwSetFramebufferSizeCallback(
        srcHandle, [](auto srcWindow, auto width, auto height) {
            auto ptr = (CallbackData *)glfwGetWindowUserPointer(srcWindow);
            ptr->guiPtr->cacheData_.fbWidth = width;
            ptr->guiPtr->cacheData_.fbHeight = height;
            ptr->guiPtr->params_.frameNum = 0;
        });

    glfwSetMouseButtonCallback(
        srcHandle,
        [](auto srcWindow, int button, int action, [[maybe_unused]] int mods) {
            auto ptr = (CallbackData *)glfwGetWindowUserPointer(srcWindow);
            if (button == GLFW_MOUSE_BUTTON_LEFT)
            {
                auto &control = ptr->guiPtr->cacheData_.cameraControl;
                if (action == GLFW_PRESS)
                {
                    control.valid = true;
                    glfwGetCursorPos(srcWindow, &control.beginPos.x,
                                     &control.beginPos.y);
                }
                else if (action == GLFW_RELEASE)
                {
                    control.valid = false;
                }
            }
        });

    glfwSetCursorPosCallback(srcHandle, [](auto srcWindow, double xpos,
                                           double ypos) {
        auto ptr = (CallbackData *)glfwGetWindowUserPointer(srcWindow);
        auto &data = ptr->guiPtr->cacheData_;

        static constexpr glm::dvec2 speed{ std::numbers::pi / 1000 };
        if (data.cameraControl.valid)
        {
            auto newPos = glm::dvec2{ xpos, ypos },
                 delta = newPos - data.cameraControl.beginPos;
            if (delta == glm::dvec2{ 0.0 }) // No need to change camera position
                return;

            delta *= speed;
            data.cameraSphCoord += glm::vec3{ -delta.y, delta.x, 0.0f };
            data.cameraControl.beginPos = newPos;
            ptr->guiPtr->params_.frameNum = 0;
        }
    });

    glfwSetScrollCallback(
        srcHandle,
        [](auto srcWindow, [[maybe_unused]] double xoffset, double yoffset) {
            static constexpr double speed = 0.01;

            auto ptr = (CallbackData *)glfwGetWindowUserPointer(srcWindow);
            auto &cameraSphCoord = ptr->guiPtr->cacheData_.cameraSphCoord;
            auto newZ = cameraSphCoord.z - static_cast<float>(speed * yoffset);
            if (cameraSphCoord.z < 0)
                cameraSphCoord.z = std::min(newZ, -0.1f);
            else
                cameraSphCoord.z = std::max(newZ, 0.1f);
            ptr->guiPtr->params_.frameNum = 0;
        });
}

void GUI::LoadConfig(const std::filesystem::path &path)
{
    ConfiguationLoader::Load(path, params_);
    SyncCache();
}

void GUI::SaveConfig(const std::filesystem::path &path) const
{
    ConfiguationLoader::Save(path, params_);
}
