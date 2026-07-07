/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
#include <cstdint>
#include <vector>

#include "Utils.hpp"

struct Image
{
    float *data;
    std::uint32_t width, height, channelNum;

    auto Size() const { return static_cast<std::size_t>(width) * height; }
};

class PiecewiseConstant2D
{
    friend class GPUPiecewiseConstant2D;

public:
    PiecewiseConstant2D() = default;
    PiecewiseConstant2D(Image image, float *total = nullptr);
    struct SampleResult
    {
        glm::vec2 xy;
        float pdf;
    };

    SampleResult Sample(float u1, float u2) const;
    bool Valid() const noexcept
    { // Default constructed / moved ones may have 0 size.
        return marginalCDF_.size() > 0 && conditionalCDF_.size() > 0;
    }

    std::uint32_t GetWidth() const noexcept { return width_; }
    std::uint32_t GetHeight() const noexcept
    {
        return static_cast<std::uint32_t>(marginalCDF_.size() - 1);
    }

private:
    std::uint32_t width_;               // For quick indexing.
    std::vector<float> marginalCDF_;    // [0..height]
    std::vector<float> conditionalCDF_; // [height][0..width]
};

class GPUPiecewiseConstant2D
{
public:
    using SampleResult = PiecewiseConstant2D::SampleResult;

    GPUPiecewiseConstant2D(PiecewiseConstant2D &);
    struct Proxy
    {
        std::uint32_t width_, height_;
        float *marginalCDF_;
        float *conditionalCDF_;

        __device__ SampleResult Sample(float u1, float u2) const;
    };

    auto GetProxy() const
    {
        return Proxy{ width_, height_, marginalCDF_.Get(),
                      conditionalCDF_.Get() };
    }

private:
    std::uint32_t width_, height_;
    UniqueResource<float *, CUDADeleter> marginalCDF_;
    UniqueResource<float *, CUDADeleter> conditionalCDF_;
};
