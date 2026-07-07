/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
#ifdef __CUDACC__

#include "Utils.hpp"

template<typename T>
__device__ T Mix(T x, T y, T a)
{
    return x * ((T)1.0 - a) + y * a;
}

template<typename T, typename U>
__device__ void Get2DFeature(glm::vec2 uv, const T *encoding, glm::ivec2 res,
                             std::size_t totalSize, unsigned int dim,
                             U *featureOut)
{
    assert(glm::all(glm::greaterThanEqual(uv, glm::vec2{ -0.001f })));
    assert(glm::all(glm::lessThanEqual(uv, glm::vec2{ 1.001f })));

    glm::vec2 scaledIdx = uv * glm::vec2{ res - 1 } + 0.5f;

    glm::ivec2 baseIdx{ scaledIdx };
    auto frac = scaledIdx - glm::vec2{ baseIdx };

    std::size_t stride0 = 1, stride1 = res.x;

    std::size_t idx0 = baseIdx.x + baseIdx.y * stride1;
    std::size_t indices[4] = { idx0 % totalSize, (idx0 + stride0) % totalSize,
                               (idx0 + stride1) % totalSize,
                               (idx0 + stride1 + stride0) % totalSize };

    for (auto &idx : indices)
        idx *= dim;

    for (std::size_t i = 0; i < dim; i++)
    {
        T a = Mix<T>(encoding[indices[0] + i], encoding[indices[1] + i],
                     frac.x),
          b = Mix<T>(encoding[indices[2] + i], encoding[indices[3] + i],
                     frac.x);
        featureOut[i] = static_cast<U>(Mix<T>(a, b, frac.y));
    }
}

template<typename T, typename U>
__device__ void Get3DFeature(glm::vec3 voxel, const T *encoding, glm::ivec3 res,
                             std::size_t totalSize, unsigned int dim,
                             U *featureOut)
{
    assert(glm::all(glm::greaterThanEqual(voxel, glm::vec3{ -0.001f })));
    assert(glm::all(glm::lessThanEqual(voxel, glm::vec3{ 1.001f })));

    glm::vec3 scaledIdx = voxel * glm::vec3{ res - 1 } + 0.5f;

    glm::ivec3 baseIdx{ scaledIdx };
    auto frac = scaledIdx - glm::vec3{ baseIdx };
    std::size_t stride0 = 1, stride1 = res.x, stride2 = res.x * res.y;

    std::size_t idx0 = baseIdx.x + baseIdx.y * stride1 + baseIdx.z * stride2;
    std::size_t indices[8] = { idx0 % totalSize,
                               (idx0 + stride0) % totalSize,
                               (idx0 + stride1) % totalSize,
                               (idx0 + stride1 + stride0) % totalSize,
                               (idx0 + stride2) % totalSize,
                               (idx0 + stride2 + stride0) % totalSize,
                               (idx0 + stride2 + stride1) % totalSize,
                               (idx0 + stride2 + stride1 + stride0) %
                                   totalSize };

    for (auto &idx : indices)
        idx *= dim;

    for (std::size_t i = 0; i < dim; i++)
    {
        T a = Mix<T>(encoding[indices[0] + i], encoding[indices[1] + i],
                     frac.x),
          b = Mix<T>(encoding[indices[2] + i], encoding[indices[3] + i],
                     frac.x),
          c = Mix<T>(encoding[indices[4] + i], encoding[indices[5] + i],
                     frac.x),
          d = Mix<T>(encoding[indices[6] + i], encoding[indices[7] + i],
                     frac.x);

        T aa = Mix<T>(a, b, frac.y), bb = Mix<T>(c, d, frac.y);
        featureOut[i] = static_cast<U>(Mix<T>(aa, bb, frac.z));
    }
}

template<int InputDim, typename T, typename U>
__device__ void GetFeature(glm::vec<InputDim, float> voxel, const T *encoding,
                           glm::vec<InputDim, int> res, std::size_t totalSize,
                           unsigned int dim, U *featureOut)
{
    if constexpr (InputDim == 2)
        Get2DFeature(voxel, encoding, res, totalSize, dim, featureOut);
    else if constexpr (InputDim == 3)
        Get3DFeature(voxel, encoding, res, totalSize, dim, featureOut);
    else
        static_assert(false, "Currently only 2/3D sampling is supported.");
}

template<typename T>
__forceinline__ __device__ T Maximum(T a, T b)
{
    if constexpr (std::is_same_v<T, half>)
        return __hmax(a, b);
    else
        return max(a, b);
}

template<typename T>
__forceinline__ __device__ T Minimum(T a, T b)
{
    if constexpr (std::is_same_v<T, half>)
        return __hmin(a, b);
    else
        return min(a, b);
}

#endif