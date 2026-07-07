/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "render/Volume.cuh"

__global__ void SampleTest(TrMapKernelData map, VolumeKernelData data)
{
    for (int i = 0; i < 4; i++)
        printf("%f ",
               map.map.Sample(0.3, 0.2, 0.1, i) * data.GetInvMaxDensity());
    return;
}

int main(int argc, const char *argv[])
{
    if (argc != 2)
        return 1;

    GPUSimpleVolume volume{ argv[1], 4, { 1024, 1024, 1024 } };

    auto trMap = TransmittanceMap{ *volume.Get(), 3, 4 };
    trMap.Update(*volume.Get(), glm::vec3{ 0, 0.6, 0.8 }, 0);
    cudaDeviceSynchronize();
    SampleTest<<<1, 1>>>(TrMapKernelData{ *trMap.Get() },
                         VolumeKernelData{ *volume.Get() });
    cudaDeviceSynchronize();
    return 0;
}