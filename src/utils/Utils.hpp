/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <glm/glm.hpp>

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

void LogError(const char *errMsg);
void LogError(std::string_view errMsg);

void CheckCUDAError(cudaError_t err);

#define LAZY_STR(x) [&]() { return x; }
template<bool LogOnly = false, typename U = void>
void CheckError(bool success, U &&strGen)
{
    if (!success)
    {
        if constexpr (std::is_invocable_v<decltype(strGen)>)
        {
            LogError(strGen());
        }
        else
        {
            LogError(strGen);
        }

        if constexpr (!LogOnly)
            throw std::runtime_error{ "Error" };
    }
}

template<typename T1, typename T2>
inline constexpr auto RoundUpDiv(T1 a, T2 b)
{
    using T = std::common_type_t<T1, T2>;
    return (T{ a } - T{ 1 }) / T{ b } + T{ 1 };
}

std::ostringstream ReadAllFromFile(const std::filesystem::path &path,
                                   bool addTrailingZero = true,
                                   bool isBinary = false);

template<typename T, typename Deleter, auto Default = T{},
         bool CheckDefault = false>
class UniqueResource
{
    T res_{ Default };

    void Destroy_() noexcept
    {
        if constexpr (CheckDefault)
        {
            if (res_ == Default)
                return;
        }
        Deleter{}(res_);
    }

public:
    UniqueResource() = default;

    template<typename... Args>
    explicit UniqueResource(Args &&...args)
        : res_{ std::forward<Args>(args)... }
    {
    }

    UniqueResource(UniqueResource &&another) noexcept
        : res_{ std::exchange(another.res_, Default) }
    {
    }
    UniqueResource &operator=(UniqueResource &&another) noexcept
    {
        if (this == &another)
            return *this;

        Destroy_();
        res_ = std::exchange(another.res_, Default);
        return *this;
    }

    T Release() noexcept { return std::exchange(res_, Default); }
    void Reloc(const T &elem) { res_ = elem; }
    void Reloc(T &&elem) { res_ = std::move(elem); }
    void Reset(const T &elem)
    {
        Destroy_();
        res_ = elem;
    }
    void Reset(T &&elem = Default)
    {
        Destroy_();
        res_ = std::move(elem);
    }
    // Only to fulfill standard library.
    friend void swap(UniqueResource &a, UniqueResource &b) noexcept
    {
        std::ranges::swap(a.res_, b.res_);
    }
    void Swap(UniqueResource &another) { swap(*this, another); }

    const T &Get() const noexcept { return res_; }

    ~UniqueResource() { Destroy_(); }
};

template<auto FuncPtr>
struct Deleter
{
    template<typename T>
    void operator()(T resource) const
    {
        FuncPtr(resource);
    }
};

using CUDADeleter = Deleter<cudaFree>;
using CUDAArrayDeleter = Deleter<cudaFreeArray>;
using CUDAMipmappedArrayDeleter = Deleter<cudaFreeMipmappedArray>;

using CUDATextureDeleter = Deleter<cudaDestroyTextureObject>;
using CUDASurfaceDeleter = Deleter<cudaDestroySurfaceObject>;

// Code below is adjusted from PBRT-v4.
struct Ray
{
    glm::vec3 ori, dir;
};

// All(pMin < pMax) is assumed to be true.
class Bounds3f
{
    glm::vec3 pMin_, pMax_;

public:
    __host__ __device__ Bounds3f(glm::vec3 pMin, glm::vec3 pMax)
        : pMin_{ pMin }, pMax_{ pMax }
    {
    }

    __host__ __device__ glm::vec3 Offset(glm::vec3 p) const
    {
        return (p - pMin_) / (pMax_ - pMin_);
    }

    __host__ __device__ glm::vec3 Center() const
    {
        return pMin_ + (pMax_ - pMin_) / 2.0f;
    }
    __host__ __device__ glm::vec3 Diagonal() const { return pMax_ - pMin_; }
    __host__ __device__ glm::vec3 Min() const { return pMin_; }
    __host__ __device__ glm::vec3 Max() const { return pMax_; }
    __host__ __device__ bool Include(glm::vec3 p) const
    {
        return glm::all(glm::lessThanEqual(pMin_, p)) &&
               glm::all(glm::lessThanEqual(p, pMax_));
    }

    __host__ __device__ bool IntersectP(glm::vec3 o, glm::vec3 d, float tMax,
                                        float *hitt0, float *hitt1) const
    {
        float t0 = 0, t1 = tMax;
        for (int i = 0; i < 3; ++i)
        {
            // Update interval for _i_th bounding box slab
            float invRayDir = 1 / d[i];
            float tNear = (pMin_[i] - o[i]) * invRayDir;
            float tFar = (pMax_[i] - o[i]) * invRayDir;
            // Update parametric interval from slab intersection $t$ values
            if (tNear > tFar)
            {
                auto temp = tNear;
                tNear = tFar, tFar = temp;
            }
            // Update _tFar_ to ensure robust ray--bounds intersection
            // tFar *= 1 + 2 * gamma(3);

            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;
            if (t0 > t1)
                return false;
        }
        if (hitt0)
            *hitt0 = t0;
        if (hitt1)
            *hitt1 = t1;
        return true;
    }

    __host__ __device__ float IntersectUnchecked(glm::vec3 o, glm::vec3 invDir,
                                                 glm::ivec3 dirIsPos) const
    {
        glm::vec3 bounds[2] = { pMin_, pMax_ };
        float tMax[3] = { (bounds[dirIsPos[0]].x - o.x) * invDir.x,
                          (bounds[dirIsPos[1]].y - o.y) * invDir.y,
                          (bounds[dirIsPos[2]].z - o.z) * invDir.z };

        for (auto &f : tMax)
        {
            f = !(f >= 0) ? INFINITY : f;
        }

        return glm::min(glm::min(tMax[0], tMax[1]), tMax[2]);
    }
};