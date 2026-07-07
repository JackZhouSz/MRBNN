/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "render/Network.hpp"

#include <filesystem>
#include <fstream>

static constexpr int inputDim = 44;
using TestFloatType = half;

TestFloatType input[inputDim]{
    0.4453, 0.6216, 0.9590, 0.4836, 0.1464, 0.9595, 0.6636, 0.1217, 0.2213,
    0.5933, 0.5498, 0.2281, 0.1527, 0.1672, 0.8213, 0.8862, 0.7686, 0.3545,
    0.2881, 0.8389, 0.3064, 0.5786, 0.2893, 0.7397, 0.4021, 0.4456, 0.9087,
    0.9365, 0.8223, 0.2761, 0.7197, 0.7480, 0.9546, 0.4739, 0.2754, 0.2443,
    0.4424, 0.3840, 0.3911, 0.3604, 0.6172, 0.1433, 0.5898, 0.4080
};

__device__ TestFloatType deviceInput[inputDim];

__global__ void InferenceTest(NetworkGPUProxy<TestFloatType> network)
{
    network.PrintRepl();
    TestFloatType outputBuffer[inputDim]{};
    network.Inference(deviceInput, outputBuffer);
    for (int i = 0; i < network.GetOutputDim(); i++)
        printf("%f ", (float)Maximum(outputBuffer[i], NetworkFloat{ 0 }));
    return;
}

// See data/test/network.test.txt
void TestNetwork(const std::filesystem::path &rootPath,
                 const nlohmann::json &config)
{
    auto network =
        Network<TestFloatType>{ rootPath / "final_mlp", config["network_final"],
                                inputDim, 1 };
    InferenceTest<<<1, 1>>>(network.ToProxy());
    cudaDeviceSynchronize();
}

template<int N, typename T>
__global__ void EncodingSampleTest(EncodingGPUProxy<TestFloatType> network,
                                   T coord)
{
    TestFloatType outputBuffer[8];
    for (int i = 0; i < 4; i++)
    {
        network.Sample<N>(outputBuffer, 8, i, coord);
        for (int i = 0; i < 8; i++)
            printf("%f ", (float)outputBuffer[i]);
    }
}

void TestEncoding(const std::filesystem::path &rootPath,
                  const nlohmann::json &config)
{
    auto encoding = Encoding<TestFloatType>{ rootPath / "view.bin",
                                             config["view_encoding"], 2 };
    EncodingSampleTest<2><<<1, 1>>>(encoding.ToProxy(), glm::vec2{ 0.1, 0.2 });
    cudaDeviceSynchronize();
}

__global__ void SHEncodingSampleTest(
    SHEncodingGPUProxy<TestFloatType, 2> network)
{
    TestFloatType outputBuffer[8];
    for (int i = 0; i < 4; i++)
    {
        network.Sample<3>(outputBuffer, 8, i, glm::vec3{ 0.1, 0.2, 0.3 },
                          glm::vec3{ 0, 0, 1 }, 0.857f);
        for (int i = 0; i < 8; i++)
            printf("%f ", (float)outputBuffer[i]);
    }
}

// 0.061310 -0.001369 0.026352 0.053467 -0.056244 -0.084473 0.047577 -0.001521
void TestSHEncoding(const std::filesystem::path &rootPath,
                    const nlohmann::json &config)
{
    auto encoding =
        SphericalEncoding<TestFloatType, 2>{ rootPath / "ms",
                                             config["ms_encoding"], 3 };
    SHEncodingSampleTest<<<1, 1>>>(encoding.ToProxy());
    cudaDeviceSynchronize();
}

void Test3DEncoding(const std::filesystem::path &rootPath,
                    const nlohmann::json &config)
{
    auto encoding = Encoding<TestFloatType>{ rootPath / "ms0.bin",
                                             config["ms_encoding"], 3 };
    EncodingSampleTest<3>
        <<<1, 1>>>(encoding.ToProxy(), glm::vec3{ 0.1, 0.2, 0.3 });
    cudaDeviceSynchronize();
}

int main(int argc, const char *argv[])
{
    if (argc != 2)
        return 1;

    CheckCUDAError(cudaMemcpyToSymbol(deviceInput, input, sizeof(input)));

    std::filesystem::path rootPath{ argv[1] };
    std::ifstream fin{ rootPath / "config.json" };
    auto config = nlohmann::json::parse(fin);

    TestNetwork(rootPath, config);
    TestEncoding(rootPath, config);
    Test3DEncoding(rootPath, config);
    TestSHEncoding(rootPath, config);

    return 0;
}