/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
#include "Utils.hpp"

class MIPMap3D
{
    friend class MIPMap3DView;

protected:
    UniqueResource<cudaMipmappedArray_t, CUDAMipmappedArrayDeleter> mipmap_;
    UniqueResource<cudaTextureObject_t, CUDATextureDeleter> tex_;

public:
    MIPMap3D() = default;
    explicit operator bool() const noexcept { return mipmap_.Get() != nullptr; }
    MIPMap3D(float *baseVoxels, int resX, int resY, int resZ, int mipmapLevel,
             bool innerLinear = false, bool outerLinear = false,
             int channelNum = 1);
    int RecreateTexture(int expectedLevel, bool innerLinear = false,
                        bool outerLinear = false, bool cachedLevel = false);
};

class MutableMIPMap3D : public MIPMap3D
{
    class Cache
    {
        UniqueResource<cudaArray_t, CUDAArrayDeleter> mem_;
        UniqueResource<cudaSurfaceObject_t, CUDASurfaceDeleter> surf_;
        cudaArray_t memInMIPMap_;

    public:
        Cache(cudaArray_t mem, cudaArray_t memInMIPMap);
        cudaArray_t GetSrcMemory() const noexcept { return mem_.Get(); }
        cudaArray_t GetDstMemory() const noexcept { return memInMIPMap_; }
        cudaSurfaceObject_t GetSurface() const noexcept { return surf_.Get(); }
    };

    std::vector<Cache> updateCache_;
    cudaExtent resolution_;

public:
    MutableMIPMap3D() = default;
    MutableMIPMap3D(int resX, int resY, int resZ, int mipmapLevel,
                    bool innerLinear = false, bool outerLinear = false);
    void Update(float *baseVoxels) const;
    void Sync() const;
    cudaExtent GetResolution() const noexcept { return resolution_; }
};

class MIPMap3DView
{
    cudaTextureObject_t tex_;

public:
    MIPMap3DView(const MIPMap3D &m) noexcept : tex_{ m.tex_.Get() } {}
    // WARNING: Actually not documented that 0 represents an invalid state, but
    // seems to work.
    __host__ __device__ explicit operator bool() const noexcept
    {
        return tex_ != 0;
    }
#ifdef __CUDACC__
    template<typename T = float>
    __device__ auto Sample(float x, float y, float z, float lod) const
    {
        return tex3DLod<T>(tex_, x, y, z, lod);
    }
#endif
};