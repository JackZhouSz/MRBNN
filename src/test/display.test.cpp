/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "display/Window.hpp"

#include "display/GLInterOp.hpp"

int main()
{
    auto &mainWindow = MainWindow::GetInstance();
    GLInterOpDisplay displayBuffer;
    std::vector<float> colorBuffer;

    while (!mainWindow.ShouldClose())
    {
        mainWindow.BeginNewFrame();
        auto size = mainWindow.GetWidthAndHeight();
        auto width = size.x, height = size.y;
        displayBuffer.TryResize((unsigned int)width, (unsigned int)height);
        colorBuffer.resize(((std::size_t)width * height * 4));

        float delta = 1.0f / (height * width);
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
                std::size_t base = i * width + j;
                float color = base * delta;
                base *= 4;
                colorBuffer[base] = colorBuffer[base + 1] =
                    colorBuffer[base + 2] = colorBuffer[base + 3] = color;
            }

        {
            auto ptr = displayBuffer.GetDevicePtrGuard();
            cudaMemcpy(ptr.GetDevicePtr(), colorBuffer.data(),
                       colorBuffer.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        displayBuffer.Draw();

        mainWindow.EndCurrentFrame();
        mainWindow.NewEvents();
    }

    return 0;
}