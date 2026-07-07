/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "display/FileDialog.hpp"
#include "display/GLInterOp.hpp"
#include "display/Window.hpp"

#include "gui/GUI.hpp"

#include <chrono>
#include <format>
#include <iostream>
#include <optional>

template<typename T = decltype([](float fps) {
             std::cerr << std::format("\rFPS: {}", fps);
         })>
class Profiler
{
    std::chrono::nanoseconds duration_;
    int cnt_ = 0;
    T process_;

    class Probe
    {
        Profiler &profiler_;
        int limit_;
        std::chrono::steady_clock::time_point begin_;

    public:
        Probe(Profiler &profiler, int limit)
            : profiler_{ profiler }, limit_{ limit },
              begin_{ std::chrono::steady_clock::now() }
        {
        }
        ~Probe()
        {
            auto end = std::chrono::steady_clock::now();
            profiler_.duration_ += (end - begin_);

            if (++profiler_.cnt_ >= limit_)
                profiler_.BackZero_();
        }
    };

    void BackZero_()
    {
        using namespace std::literals;
        auto fps = static_cast<float>((1.0s * cnt_) / duration_);
        process_(fps);
        duration_ = duration_.zero();
        cnt_ = 0;
    }

public:
    Profiler(T process = {}) : process_{ std::move(process) } {}
    Probe EmitProbe(int limit) { return Probe{ *this, limit }; }
};

static void TrySetNewInterface(auto &gui, auto &interface)
{
    auto &newPath = gui.TryGetNewWorkDir();
    if (!newPath)
        return;
    try
    {
        interface.emplace(*newPath);
        if (auto configPath = *newPath / "init.json";
            std::filesystem::exists(configPath))
        {
            gui.LoadConfig(configPath);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << std::format("Error when opening working directory: {}\n",
                                 ex.what());
    }
}

int main()
{
    COMContextGuard _; // To use file dialog on windows.
    auto &mainWindow = MainWindow::GetInstance();
    std::optional<RenderInterfaceWithTCNN> interface;
    GLInterOpDisplay radianceBuffer{ GLInterOpDisplay::PixelFormat::Float3 };

    // This must be after the radianceBuffer is created, to make its texture
    // attached to main window context.
    auto guiWindow = WindowForImGui(500, 500);
    GUI gui;
    gui.RegisterCallbacks(guiWindow, mainWindow);
    // To trigger callback immediately.
    mainWindow.SetPosition(mainWindow.GetPosition() + glm::ivec2{ 1, 1 });

    auto &frameBufferInfo = gui.GetFrameBufferInfo();
    frameBufferInfo.channelNum = 3;

    auto showFPSInGUI = [&gui](float fps) { gui.SetFPS(fps); };
    Profiler profiler{ std::ref(showFPSInGUI) };
    while (!mainWindow.ShouldClose())
    {
        auto _1 = profiler.EmitProbe(100);
        auto _2 = EventGuard{};

        { // Draw ImGUI window
            auto _3 = WindowGuard{ guiWindow };
            auto fitSize = glm::ivec2{ glm::ceil(gui.Draw()) };
            if (guiWindow.GetWindowWidthAndHeight() != fitSize)
                guiWindow.Resize(fitSize);
            TrySetNewInterface(gui, interface);
        }

        { // Draw main window
            auto _3 = WindowGuard{ mainWindow };
            auto size = mainWindow.GetWidthAndHeight();
            auto width = size.x, height = size.y;
            if (width == 0 || height == 0 || !interface.has_value())
                continue;

            if (const auto &path = gui.TryGetNewSkyboxBakingWorkDir())
                interface->EquipEnvBaking(&path.value());

            radianceBuffer.TryResize(height, width);
            frameBufferInfo.size = size;
            {
                auto devicePtrGuard = radianceBuffer.GetDevicePtrGuard();
                frameBufferInfo.bufferPtr =
                    static_cast<float *>(devicePtrGuard.GetDevicePtr());
                interface->Render(gui.GetRenderParameters());
                cudaDeviceSynchronize();
            }
            radianceBuffer.Draw();
        }
    }

    return 0;
}