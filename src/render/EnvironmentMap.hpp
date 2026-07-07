/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
#include "PiecewiseConstant2D.hpp"
#include "Texture2D.hpp"
#include "stbi/stb_image.h"

#include <filesystem>
#include <numbers>

// Unfortunately, we place it here for nvcc seperate compilation bug.
// (i.e. have to put Proxy::SampleTexture in header, even with -rdc=true)
template<bool AutoWrap = false>
__host__ __device__ glm::vec2 DirectionToUV(glm::vec3 dir)
{
    float u = atan2f(dir.x, -dir.z) * (0.5f * std::numbers::inv_pi_v<float>);
    float v =
        acosf(glm::clamp(dir.y, -1.0f, 1.0f)) * std::numbers::inv_pi_v<float>;

    glm::vec2 result{ u, v };

    if constexpr (AutoWrap)
        return glm::fract(result);

    return result;
}

class EnvironmentMap
{
public:
    EnvironmentMap() = default;
    EnvironmentMap(const std::filesystem::path &hdriPath,
                   float exposure = 1.0f);
    bool Valid() const noexcept { return image_.Get() != nullptr; }
    float &Exposure() noexcept { return exposure_; }
    float Exposure() const noexcept { return exposure_; }

    struct SampleResult
    {
        glm::vec3 radiance;
        glm::vec3 dir;
        float pdf;
    };

    SampleResult SampleDirection(float u1, float u2) const;

    class Proxy
    {
        float exposure_;
        glm::vec2 deltaToMatchSampler_;
        Texture2DView tex_;

    public:
        Proxy() = default;
        Proxy(const EnvironmentMap &envMap);
        __host__ __device__ float Exposure() const { return exposure_; }
#ifdef __CUDACC__
        __device__ glm::vec3 SampleTexture(glm::vec3 dir) const
        {
            if (!tex_)
                return glm::vec3{ 0 };

            auto uv = DirectionToUV(dir);
            uv += deltaToMatchSampler_;
            auto result = tex_.Sample<float4>(uv.x, uv.y);
            return glm::vec3{ result.x, result.y, result.z } * exposure_;
        }
#endif
    };

    float Power() const noexcept { return totalIllum_ * exposure_; }
    auto GetProxy() const noexcept { return Proxy{ *this }; }
    const auto &GetSampler() const noexcept { return sampler_; }
    glm::uvec2 GetWidthAndHeight() const noexcept
    {
        return { sampler_.GetWidth(), sampler_.GetHeight() };
    }

private:
    glm::vec3 SampleImage_(glm::vec2 xy) const;

    float exposure_ = 0.0f;
    std::uint32_t channelNum_ = 0;
    // Well, because stbi doesn't provide "load_to_buffer"
    float totalIllum_ = 0.0f;
    UniqueResource<float *, Deleter<stbi_image_free>> image_;
    PiecewiseConstant2D sampler_;

    Texture2D tex_;
};