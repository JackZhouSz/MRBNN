/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
#include "Utils.hpp"
#include <filesystem>

class Texture2D
{
    friend class Texture2DView;

    UniqueResource<cudaArray_t, CUDAArrayDeleter> buffer_;
    UniqueResource<cudaTextureObject_t, CUDATextureDeleter> tex_;

    template<typename T>
    void LoadResources_(const std::filesystem::path &);
    template<typename T>
    void LoadResources_(T *ptr, int width, int height, int channelNum);

public:
    enum class Format
    {
        UByte,
        UShort,
        Float
    };

    Texture2D() = default;
    Texture2D(const std::filesystem::path &path, Format format = Format::Float);
    Texture2D(void *ptr, int width, int height, int channelNum,
              Format format = Format::Float);
};

class MIPMap2D
{
    friend class Texture2DView;

    UniqueResource<cudaMipmappedArray_t, CUDAMipmappedArrayDeleter> mipmap_;
    UniqueResource<cudaTextureObject_t, CUDATextureDeleter> tex_;
    float estimatedAveragePower_ = 0.0f;

    template<typename T>
    void LoadResources_(const std::filesystem::path &, int);

public:
    enum class Format
    {
        UByte,
        UShort,
        Float
    };

    MIPMap2D() = default;
    MIPMap2D(const std::filesystem::path &path, int mipmapLevel,
             Format format = Format::Float);
    float GetEstimatedAveragePower() const noexcept
    {
        return estimatedAveragePower_;
    }
};

class Texture2DView
{
    cudaTextureObject_t tex_;

public:
    Texture2DView(const Texture2D &m) noexcept : tex_{ m.tex_.Get() } {}
    Texture2DView(const MIPMap2D &m) noexcept : tex_{ m.tex_.Get() } {}

    __device__ explicit operator bool() const noexcept { return tex_ != 0; }
#ifdef __CUDACC__
    template<typename T>
    __device__ T Sample(float x, float y) const
    {
        return tex2D<T>(tex_, x, y);
    }

    template<typename T>
    __device__ T SampleLod(float x, float y, float lod) const
    {
        return tex2DLod<T>(tex_, x, y, lod);
    }
#endif
};