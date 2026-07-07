/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include <array>
#include <filesystem>
#include <memory>

#include "Utils.hpp"

class AlbedoVolume
{
public:
    struct Data;

private:
    friend class GPUSimpleVolume;
    std::unique_ptr<Data> impl_;

public:
    AlbedoVolume() noexcept;
    AlbedoVolume(const std::filesystem::path &path, std::size_t skipByteNum,
                 const std::array<int, 3> &init_res, int channelNum = 3);
    AlbedoVolume(const AlbedoVolume &) = delete;
    AlbedoVolume &operator=(const AlbedoVolume &) = delete;
    AlbedoVolume(AlbedoVolume &&) noexcept;
    AlbedoVolume &operator=(AlbedoVolume &&) noexcept;
    ~AlbedoVolume();
};

class GPUSimpleVolume
{
public:
    struct VolumeData;

private:
    friend class GPUSimpleVolumeView;
    std::unique_ptr<VolumeData> impl_;

public:
    GPUSimpleVolume() noexcept;
    GPUSimpleVolume(const std::filesystem::path &path, std::size_t skipByteNum,
                    const std::array<int, 3> &init_res,
                    const Bounds3f &init_bound = { { -0.5f, -0.5f, -0.5f },
                                                   { 0.5f, 0.5f, 0.5f } },
                    int mipmapLevel = 0, AlbedoVolume &&albedoVolume = {});
    GPUSimpleVolume(const GPUSimpleVolume &) = delete;
    GPUSimpleVolume &operator=(const GPUSimpleVolume &) = delete;
    GPUSimpleVolume(GPUSimpleVolume &&) noexcept; // Pimpl.
    GPUSimpleVolume &operator=(GPUSimpleVolume &&) noexcept;
    ~GPUSimpleVolume();

    const VolumeData *Get() const noexcept { return impl_.get(); }
    void RecreateTexture(bool innerLinear, bool outerLinear);
};

class TransmittanceMap
{
public:
    struct MapData;

private:
    std::unique_ptr<MapData> impl_;

public:
    TransmittanceMap() noexcept;
    TransmittanceMap(const GPUSimpleVolume::VolumeData &volumeData,
                     int beginLevel, int mipmapLevel);
    TransmittanceMap(std::array<int, 3> baseResolution, int mipmapLevel);
    TransmittanceMap(const TransmittanceMap &) = delete;
    TransmittanceMap &operator=(const TransmittanceMap &) = delete;
    TransmittanceMap(TransmittanceMap &&) noexcept;
    TransmittanceMap &operator=(TransmittanceMap &&) noexcept;
    ~TransmittanceMap();

    const MapData *Get() const noexcept { return impl_.get(); }
    void Update(const GPUSimpleVolume::VolumeData &volumeData,
                glm::vec3 lightDir, int sampleLevel, int stepNum = 0,
                bool forced = false) const;
};