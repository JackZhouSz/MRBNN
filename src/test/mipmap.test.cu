/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "render/MIPMap.hpp"
#include <chrono>
#include <iostream>
#include <vector>

__global__ void OutputMIPMap(MIPMap3DView view, int levelNum, int resX,
                             int resY, int resZ)
{
    for (int x = 0; x < resX; x++)
        for (int y = 0; y < resY; y++)
            for (int z = 0; z < resZ; z++)
            {
                printf("%d %d %d:", x, y, z);
                for (int i = 0; i <= levelNum; i++)
                {
                    printf("%f ",
                           view.Sample((x + 0.5f) / resX, (y + 0.5f) / resY,
                                       (z + 0.5f) / resZ, i));
                }
                printf("\n");
            }
}

void PerfTest()
{
    constexpr int res{ 1024 };
    std::vector<float> buffer(res * res * res);

    auto t1 = std::chrono::steady_clock::now();
    MIPMap3D mipmap{ buffer.data(), res, res, res, 9 };
    CheckCUDAError(cudaGetLastError());
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    using namespace std::literals::chrono_literals;

    std::cout << "Time: " << (t2 - t1) / 1ms << "ms\n";
}

void BasicTest()
{
    constexpr int resX = 4, resY = 4, resZ = 4;
    constexpr int size = resX * resY * resZ;
    float density[size];
    for (int i = 0; i < size; i++)
        density[i] = 0.1f * (i + 1);

    MIPMap3D mipmap{ density, resX, resY, resZ, 3 };
    OutputMIPMap<<<1, 1>>>(mipmap, 3, resX, resY, resZ);
    CheckCUDAError(cudaGetLastError());
    cudaDeviceSynchronize();
}

void MutableTest()
{
    constexpr int resX = 4, resY = 4, resZ = 4;
    constexpr int size = resX * resY * resZ;
    float density[size];
    for (int i = 0; i < size; i++)
        density[i] = 0.1f * (i + 1);

    MutableMIPMap3D mipmap{ resX, resY, resZ, 3 };
    mipmap.Update(density);
    mipmap.Sync();
    OutputMIPMap<<<1, 1>>>(mipmap, 3, resX, resY, resZ);
    CheckCUDAError(cudaGetLastError());
    cudaDeviceSynchronize();
}

int main(int argc, const char *argv[])
{
    std::cout << "----------------Basic Test----------------\n";
    BasicTest();

    std::cout << "----------------Normal Test----------------\n";
    MutableTest();

    if (argc > 1)
    {
        PerfTest();
    }

    return 0;
}