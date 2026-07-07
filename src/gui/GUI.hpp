/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include "display/Window.hpp"
#include "render/RenderInterface.hpp"

class GUI
{
    RenderParameters params_;

    struct ParamCacheData
    {
        glm::vec3 cameraSphCoord;
        glm::vec2 lightSphCoord;
        float fps = 0.0f;

        struct CameraControl
        {
            bool valid = false;
            glm::dvec2 beginPos;
        } cameraControl;

        int fbWidth, fbHeight;

        std::optional<std::filesystem::path> workDir;
        std::optional<std::filesystem::path> skyboxBakingWorkDir;
    } cacheData_;

    struct CallbackData
    {
        GLFWwindow *targetWindow;
        GUI *guiPtr;
    } callbackData_;

public:
    GUI() { SyncCache(); }

    void SyncCache();
    void SetFPS(float fps) { cacheData_.fps = fps; }
    auto &GetFrameBufferInfo() noexcept { return params_.frameBuffer; }
    const auto &GetRenderParameters() const noexcept { return params_; }
    glm::vec2 Draw();

    void RegisterCallbacks(const WindowBase &targetWindow,
                           const WindowBase &sourceWindow);
    const std::optional<std::filesystem::path> &TryGetNewWorkDir() const
    {
        return cacheData_.workDir;
    }
    const std::optional<std::filesystem::path> &TryGetNewSkyboxBakingWorkDir()
        const
    {
        return cacheData_.skyboxBakingWorkDir;
    }

    void LoadConfig(const std::filesystem::path &path);
    void SaveConfig(const std::filesystem::path &path) const;
};