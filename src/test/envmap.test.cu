/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "render/EnvironmentMap.hpp"
#include "utils/Utils.hpp"

#include <format>
#include <iostream>

static constexpr int cs_Size = 10;
__managed__ float dirs[cs_Size * 3];

void TestSample(const EnvironmentMap &envMap)
{
    // Test multiple samples
    for (int i = 0; i < cs_Size; ++i)
    {
        float u1 = static_cast<float>(i) / float{ cs_Size } + 0.05f;
        float u2 = 0.5f;
        auto result = envMap.SampleDirection(u1, u2);

        printf("Sample %d: Direction = [%f, %f, %f], PDF = "
               "%f, Radiance = [%f, %f, %f]\n",
               i, result.dir.x, result.dir.y, result.dir.z, result.pdf,
               result.radiance.x, result.radiance.y, result.radiance.z);
        for (int j = 0; j < 3; j++)
            dirs[i * 3 + j] = result.dir[j];
    }
}

__global__ void TestSampleKernel(EnvironmentMap::Proxy envMap)
{
    for (int i = 0; i < cs_Size; ++i)
    {
        glm::vec3 dir;
        for (int j = 0; j < 3; j++)
            dir[j] = dirs[i * 3 + j];
        auto result = envMap.SampleTexture(dir);
        printf("Sample %d: Dir = [%f, %f, %f], Radiance = [%f, %f, %f]\n", i,
               dir.x, dir.y, dir.z, result.x, result.y, result.z);
    }
}

int main(int argc, const char *argv[])
{
    if (argc != 2)
        return 1;

    EnvironmentMap envMap{ argv[1] };
    TestSample(envMap);
    auto proxy = envMap.GetProxy();
    TestSampleKernel<<<1, 1>>>(proxy);
    cudaDeviceSynchronize();
    return 0;
}