/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include "MIPMap.hpp"
#include "Volume.hpp"

#include <curand_kernel.h>

class VolumeKernelData
{
    glm::vec3 resolution_;
    Bounds3f bound_;
    MIPMap3DView mipmap_;
    float invMaxDensity_;
    MIPMap3DView albedoMipMap_;

    // We don't assume swizzled coord in private method.
    __device__ float GetDensityRaw_(glm::vec3 texCoord,
                                    int sampleLevel = 0) const noexcept;

public:
    VolumeKernelData(const GPUSimpleVolume::VolumeData &volumeData);
    __host__ __device__ bool HasAlbedoGrid() const noexcept
    {
        return static_cast<bool>(albedoMipMap_);
    }
    __host__ __device__ const auto &GetBound() const noexcept { return bound_; }
    __device__ auto GetInvMaxDensity() const noexcept { return invMaxDensity_; }

    // Swizzle coords for users to use volume-related textures.
    __device__ auto ToSampleCoord(glm::vec3 pos) const noexcept
    {
        auto offset = bound_.Offset(pos);
        assert(glm::all(glm::greaterThanEqual(offset, glm::vec3{ 0 })));
        assert(glm::all(glm::lessThanEqual(offset, glm::vec3{ 1.001f })));
        return glm::vec3{ offset.z, offset.y, offset.x };
    }
    __device__ auto GetResolution() const noexcept { return resolution_; }
    __device__ glm::vec3 GetAlbedo(glm::vec3 pos) const noexcept;
    __device__ glm::vec3 GetAlbedoByCoord(glm::vec3 coord) const noexcept;
    __device__ float GetDensity(glm::vec3 pos) const noexcept;
    __device__ float GetNLogTr(glm::vec3 begin, glm::vec3 end,
                               int sampleLevel) const noexcept;
    __device__ float GetNLogTrMarching(glm::vec3 begin, glm::vec3 dir,
                                       glm::vec3 invDir, glm::ivec3 dirIsPos,
                                       int sampleLevel,
                                       int stepNum) const noexcept;
    __device__ float GetEstimatedTrDelta(curandState *seed, glm::vec3 begin,
                                         glm::vec3 end) const noexcept;
    __device__ float GetEstimatedTrRatio(curandState *seed, glm::vec3 begin,
                                         glm::vec3 end) const noexcept;
    // xyz means sampled point (out of bound if sample fails);
    // w means the remaining distance to go out of volume bound.
    __device__ glm::vec4 GetSamplePoint(curandState *seed, glm::vec3 begin,
                                        glm::vec3 dir) const noexcept;
};

struct TrMapKernelData
{
    MIPMap3DView map;
    TrMapKernelData(const TransmittanceMap::MapData &mapData);
};