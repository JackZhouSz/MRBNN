/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "Network.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <span>

static float grid_scale(uint32_t level, float log2_per_level_scale,
                        uint32_t base_resolution)
{
    return exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
}

static uint32_t grid_resolution(float scale)
{
    return (uint32_t)ceilf(scale) + 1;
}

static std::size_t next_multiple(std::size_t a, std::size_t b)
{
    return RoundUpDiv(a, b) * b;
}

static std::size_t powi(std::size_t a, int b)
{
    std::size_t result = 1;
    while (b-- > 0)
        result *= a;
    return result;
}

// Functions above are basically copied from TCNN; use different name convention
// to notice that.

template<typename T, typename ByteBuffer>
static void CreateAndCopy(UniqueResource<T *, CUDADeleter> &uniqueRes,
                          ByteBuffer view)
{
    T *gpuBuffer;
    CheckCUDAError(cudaMalloc(&gpuBuffer, view.size()));
    uniqueRes.Reloc(gpuBuffer);
    CheckCUDAError(cudaMemcpy(gpuBuffer, view.data(), view.size(),
                              cudaMemcpyHostToDevice));
}

template<typename T>
Encoding<T>::Encoding(const std::filesystem::path &path,
                      const nlohmann::json &config, int inputDims)
    : inputDims_{ inputDims }
{
    std::cout << "Reading encoding at " << path << "\n";
    auto buffer = ReadAllFromFile(path, false, true);
    auto view = buffer.view();
    CreateAndCopy(data_, view);

    int levels = config["n_levels"].get<int>();
    auto baseResolution = config["base_resolution"].get<std::uint32_t>();
    auto perLevelScale = config["per_level_scale"].get<float>();
    perLevelScale = std::log2f(perLevelScale);
    auto featurePerLevel = config["n_features_per_level"].get<int>();

    if (levels > s_maxLevels)
    {
        throw std::runtime_error{ std::format("Only support level less than {}",
                                              s_maxLevels) };
    }

    std::size_t offset = 0;
    for (int i = 0; i < levels; i++)
    {
        std::size_t currResolution =
            grid_resolution(grid_scale(i, perLevelScale, baseResolution));
        std::size_t paramsInLevel =
            next_multiple(powi(currResolution, inputDims), 8);

        resolution_[i] = currResolution;
        offsets_[i] = offset;
        offset += paramsInLevel;
    }
    offsets_[levels] = offset;

    if (offset * featurePerLevel * sizeof(T) != view.size())
    {
        throw std::runtime_error{
            "Parameter number doesn't match with configuration."
        };
    }
    featurePerLevel_ = featurePerLevel;
}

template<typename T, std::size_t Order>
SphericalEncoding<T, Order>::SphericalEncoding(
    const std::filesystem::path &rootPath, const nlohmann::json &config,
    int inputDims)
    : baseEncoding_{ std::filesystem::path{ rootPath }.concat("0.bin"), config,
                     inputDims }
{
    std::cout << "Reading SH encoding at " << rootPath << "\n";

    for (int i = 1; i < s_shNum; i++)
    {
        auto path =
            std::filesystem::path{ rootPath }.concat(std::format("{}.bin", i));
        auto buffer = ReadAllFromFile(path, false, true);
        auto view = buffer.view();
        // Inconsistent size.
        if (view.size() != baseEncoding_.offsets_.back() *
                               baseEncoding_.featurePerLevel_ * sizeof(T))
        {
            throw std::runtime_error{
                "Spherical encoding should be of the same size."
            };
        }

        CreateAndCopy(highOrderEncodings_[i - 1], view);
    }
}

template<typename T>
Network<T>::Network(const std::filesystem::path &rootPath,
                    const nlohmann::json &config, int inputDim, int outputDim)
    : layerNum_{ config["n_hidden_layers"].get<int>() }, inputDim_{ inputDim },
      hiddenDim_{ config["n_neurons"].get<int>() }, outputDim_{ outputDim }
{
    std::cout << "Reading networks at " << rootPath << "\n";

    std::size_t totalSize =
        (inputDim_ * hiddenDim_ + hiddenDim_ * hiddenDim_ * (layerNum_ - 1) +
         hiddenDim_ * outputDim_) *
        sizeof(T);

    auto CheckSize = [this](int layer, std::size_t size) {
        if (layer == 0)
            return size == inputDim_ * hiddenDim_ * sizeof(T);
        else if (layer == layerNum_)
            return size == hiddenDim_ * outputDim_ * sizeof(T);
        return size == hiddenDim_ * hiddenDim_ * sizeof(T);
    };
    auto gatherBuffer = std::make_unique_for_overwrite<std::byte[]>(totalSize);
    auto gatherPtr = gatherBuffer.get();

    for (int i = 0; i <= layerNum_; i++)
    { // This can be optimized to read into weight_ directly by spanstream in
      // C++23.
        auto buffer = ReadAllFromFile(std::filesystem::path{ rootPath }.concat(
                                          std::format(".weight{}.bin", i)),
                                      false, true);
        auto view = buffer.view();
        auto size = view.size();
        if (!CheckSize(i, size))
        {
            throw std::runtime_error{
                "MLP weight size doesn't match configuration."
            };
        }
        memcpy(gatherPtr, view.data(), size);
        gatherPtr += size;
    }

    CreateAndCopy(weight_, std::span{ gatherBuffer.get(), totalSize });

    if (auto biasPath = std::filesystem::path{ rootPath }.concat(".bias0.bin");
        std::filesystem::exists(biasPath))
    {
        auto biasBuffer = ReadAllFromFile(biasPath, false, true);
        auto view = biasBuffer.view();
        if (view.size() != hiddenDim_ * sizeof(T))
        {
            throw std::runtime_error{
                "MLP bias size doesn't match configuration."
            };
        }
        CreateAndCopy(bias_, view);
    }

    return;
}

template class Encoding<half>;
template class Encoding<float>;

// Provide more if needed.
template class SphericalEncoding<half, 2>;
template class SphericalEncoding<float, 2>;

template class Network<half>;
template class Network<float>;
