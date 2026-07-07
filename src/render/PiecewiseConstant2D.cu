/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "PiecewiseConstant2D.hpp"

#include "cuda_runtime.h"

#include <iostream>
#include <stdexcept>

namespace
{
float GetLuminance(const Image &image, std::size_t idx)
{
    if (image.channelNum != 1 && image.channelNum != 3 && image.channelNum != 4)
    {
        throw std::invalid_argument{
            "Luminance extraction of PiecewiseConstant2D only supports 1, 3 or "
            "4 channels."
        };
    }

    if (image.channelNum == 1)
        return std::abs(image.data[idx]);

    const float coeffs[] = { 0.2126f, 0.7152f, 0.0722f };
    auto baseIdx = idx * image.channelNum;
    float lum = 0.0f;
    for (std::size_t i = 0; i < 3; i++)
        lum += std::abs(image.data[baseIdx + i]) * coeffs[i];
    return lum;
}

__host__ __device__ float Sample1D(const float *cdf, std::uint32_t len, float u,
                                   float &pdf)
{
    // Theoretically total should be 1.0, but rounding error may make it
    // slightly different, so we re-normalize it here.
    float total = cdf[len], target = u * total;
    std::uint32_t low = 0, high = len;
    while (high > low + 1)
    {
        auto mid = low + ((high - low) >> 1);
        if (cdf[mid] <= target)
            low = mid;
        else
            high = mid;
    }

    float cdfLow = cdf[low], cdfHigh = cdf[low + 1];
    float prob = cdfHigh - cdfLow, offset = (target - cdfLow) / prob;
    pdf = prob * len / total;

    // Well, actually a safer way is static_cast<double>(low) + cdfLow
    // to avoid non-representable int32_t in float.
    return low + cdfLow;
}

__host__ __device__ glm::vec2 Sample2D(const float *marginalCDF,
                                       std::uint32_t len1, float u1,
                                       const float *conditionalCDF,
                                       std::uint32_t len2, float u2, float *pdf)
{
    float marginalPDF, conditionalPDF;
    float u = Sample1D(marginalCDF, len1, u1, marginalPDF);
    conditionalCDF += static_cast<std::size_t>(u) * (len2 + 1);
    float v = Sample1D(conditionalCDF, len2, u2, conditionalPDF);
    if (pdf)
        *pdf = marginalPDF * conditionalPDF;
    return { u, v };
}

} // namespace

// Allocate one more for every row; e.g. for one element, cdf needs to store
// 0.0f and 1.0f.
PiecewiseConstant2D::PiecewiseConstant2D(Image image, float *totalPtr)
    : width_{ image.width }, marginalCDF_(image.height + 1),
      conditionalCDF_(image.Size() + image.height)
{
    auto FillCDF = [](float *dst, std::uint32_t len, float total) {
        if (total == 0.0f)
        { // Uniform sampling for all values are 0.
            for (std::uint32_t j = 1; j <= len; j++)
                dst[j] = static_cast<float>(j) / len;
        }
        else
        {
            for (std::uint32_t j = 1; j <= len; j++)
                dst[j] /= total;
        }
    };

    float total = 0.0f;
    std::size_t idx = 0;
    for (std::uint32_t i = 0; i < image.height; i++)
    {
        float rowTotal = 0.0f;
        auto cdfPtr = conditionalCDF_.data() + idx + i;
        for (std::uint32_t j = 0; j < image.width; j++, idx++)
        {
            float lum = GetLuminance(image, idx);
            rowTotal += lum;
            cdfPtr[j + 1] = rowTotal;
        }
        // Calculate conditional CDF for current row.
        FillCDF(cdfPtr, image.width, rowTotal);
        total += rowTotal;
        marginalCDF_[i + 1] = total;
    }

    // Calculate marginal CDF.
    FillCDF(marginalCDF_.data(), image.height, total);
    if (totalPtr)
        *totalPtr = total;
}

auto PiecewiseConstant2D::Sample(float u1, float u2) const -> SampleResult
{
    assert(Valid());
    SampleResult result;
    result.xy = Sample2D(marginalCDF_.data(), marginalCDF_.size() - 1, u1,
                         conditionalCDF_.data(), width_, u2, &result.pdf);
    return result;
}

GPUPiecewiseConstant2D::GPUPiecewiseConstant2D(PiecewiseConstant2D &cpuData)
    : width_{ cpuData.GetWidth() }, height_{ cpuData.GetHeight() }
{
    assert(cpuData.Valid());

    auto size = cpuData.marginalCDF_.size() * sizeof(float);
    float *dPtr;
    CheckCUDAError(cudaMalloc(&dPtr, size));
    marginalCDF_.Reloc(dPtr);
    CheckCUDAError(cudaMemcpy(dPtr, cpuData.marginalCDF_.data(), size,
                              cudaMemcpyHostToDevice));

    size = cpuData.conditionalCDF_.size() * sizeof(float);
    CheckCUDAError(cudaMalloc(&dPtr, size));
    conditionalCDF_.Reloc(dPtr);
    CheckCUDAError(cudaMemcpy(dPtr, cpuData.conditionalCDF_.data(), size,
                              cudaMemcpyHostToDevice));
}

__device__ auto GPUPiecewiseConstant2D::Proxy::Sample(float u1, float u2) const
    -> SampleResult
{
    SampleResult result;
    result.xy = Sample2D(marginalCDF_, height_, u1, conditionalCDF_, width_, u2,
                         &result.pdf);
    return result;
}
