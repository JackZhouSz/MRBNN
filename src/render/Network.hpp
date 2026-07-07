/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
// To maximize inference efficiency, we inline all code except for
// initialization inside header file.
#include <array>
#include <filesystem>

#include <nlohmann/json.hpp>

#include "RenderPayload.hpp"

template<typename T>
class Encoding
{
    static constexpr inline int s_maxLevels = EncodingGPUProxy<T>::s_maxLevels;

    UniqueResource<T *, CUDADeleter> data_;

    std::array<std::size_t, s_maxLevels> resolution_;
    std::array<std::size_t, s_maxLevels + 1> offsets_;
    int featurePerLevel_;
    int inputDims_;

    template<typename, std::size_t>
    friend class SphericalEncoding;

public:
    Encoding(const std::filesystem::path &path, const nlohmann::json &config,
             int inputDims);

    auto GetInputDims() const noexcept { return inputDims_; }
    auto GetFeatureDim() const noexcept { return featurePerLevel_; }
    const auto &GetResolution() const noexcept { return resolution_; }

    EncodingGPUProxy<T> ToProxy() const noexcept
    {
        EncodingGPUProxy<T> proxy;
        proxy.data_ = data_.Get();
        for (int i = 0; i < Encoding::s_maxLevels; i++)
            proxy.resolution_[i] = resolution_[i];

        for (int i = 0; i <= Encoding::s_maxLevels; i++)
            proxy.offsets_[i] = offsets_[i];
        proxy.featurePerLevel_ = featurePerLevel_;
#ifdef DEBUG
        proxy.safetyChecker_.inputDims = inputDims_;
#endif
        return proxy;
    }
};

template<typename T, std::size_t Order>
class SphericalEncoding
{
    static constexpr inline std::size_t s_shNum = (Order + 1) * (Order + 1);

    // To store only a single copy of metadata.
    Encoding<T> baseEncoding_;
    std::array<UniqueResource<T *, CUDADeleter>, s_shNum - 1>
        highOrderEncodings_;

public:
    SphericalEncoding(const std::filesystem::path &rootPath,
                      const nlohmann::json &config, int inputDims);

    auto GetFeatureDim() const noexcept
    {
        return baseEncoding_.GetFeatureDim();
    }

    SHEncodingGPUProxy<T, Order> ToProxy() const noexcept
    {
        static_assert(SHEncodingGPUProxy<T, Order>::s_shNum == s_shNum);
        SHEncodingGPUProxy<T, Order> proxy;
        proxy.data_[0] = baseEncoding_.data_.Get();

        for (std::size_t i = 1; i < s_shNum; i++)
            proxy.data_[i] = highOrderEncodings_[i - 1].Get();

        for (int i = 0; i < std::size(baseEncoding_.resolution_); i++)
            proxy.resolution_[i] = baseEncoding_.resolution_[i];

        for (int i = 0; i < std::size(baseEncoding_.offsets_); i++)
            proxy.offsets_[i] = baseEncoding_.offsets_[i];

        proxy.featurePerLevel_ = baseEncoding_.featurePerLevel_;

#ifdef DEBUG
        proxy.safetyChecker_.inputDims = baseEncoding_.inputDims_;
#endif
        return proxy;
    }
};

template<typename T>
class Network
{
    UniqueResource<T *, CUDADeleter> weight_;
    UniqueResource<T *, CUDADeleter> bias_;

    int layerNum_;
    int inputDim_;
    int hiddenDim_;
    int outputDim_;

public:
    Network(const std::filesystem::path &rootPath, const nlohmann::json &config,
            int inputDim, int outputDim);

    NetworkGPUProxy<T> ToProxy() const noexcept
    {
        NetworkGPUProxy<T> proxy;
        proxy.weight_ = weight_.Get();
        proxy.bias_ = bias_.Get();
        proxy.layerNum_ = layerNum_;
        proxy.inputDim_ = inputDim_;
        proxy.hiddenDim_ = hiddenDim_;
        proxy.outputDim_ = outputDim_;
        return proxy;
    }
};