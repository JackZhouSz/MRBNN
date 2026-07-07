/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "Volume.cuh"

#include "vector_types.h"

#include <bit>
#include <fstream>
#include <iostream>

namespace
{
__device__ float Rand(curandState *seed)
{
    return curand_uniform(seed);
}

std::pair<std::unique_ptr<float[]>, std::size_t> ReadVoxels(
    const std::filesystem::path &path, std::size_t skipByteNum,
    const std::array<int, 3> &resolution, int desiredChannelNum = 1,
    int storedChannelNum = 1)
{
    assert(desiredChannelNum >= storedChannelNum);
    std::ifstream fin{ path, std::ios::binary };
    fin.exceptions(std::ios::failbit | std::ios::badbit);
    fin.seekg(skipByteNum);

    auto elemNum = (std::size_t)resolution[0] * resolution[1] * resolution[2];
    auto desiredElemNum = elemNum * desiredChannelNum,
         storedElemNum = elemNum * storedChannelNum;
    std::unique_ptr<float[]> voxels = std::make_unique<float[]>(desiredElemNum);
    auto rawPtr = voxels.get();
    fin.read(reinterpret_cast<char *>(rawPtr), storedElemNum * sizeof(float));

    const auto delta = desiredChannelNum - storedChannelNum;
    if (delta == 0)
        return { std::move(voxels), desiredElemNum };

    for (std::size_t desiredIdx = desiredElemNum, storedIdx = storedElemNum;
         desiredIdx > 0;
         desiredIdx -= desiredChannelNum, storedIdx -= storedChannelNum)
    {
        for (std::size_t i = 1; i <= storedChannelNum; i++)
            rawPtr[desiredIdx - delta - i] = rawPtr[storedIdx - i];
    }
    return { std::move(voxels), desiredElemNum };
}

} // namespace

struct AlbedoVolume::Data
{
    Data() = default;
    Data(const std::filesystem::path &path, std::size_t skipByteNum,
         const std::array<int, 3> &init_res, int channelNum)
    {
        CheckError(channelNum <= 4,
                   "Don't support albedo voxels with channel more than 4.");
        int desiredChannelNum = channelNum == 3 ? 4 : channelNum;

        auto [albedoGrid, _] = ReadVoxels(path, skipByteNum, init_res,
                                          desiredChannelNum, channelNum);
        mipmap = MIPMap3D{ albedoGrid.get(),
                           init_res[2],
                           init_res[1],
                           init_res[0],
                           0,
                           true,
                           false,
                           desiredChannelNum };
    }

    MIPMap3D mipmap;
};

struct GPUSimpleVolume::VolumeData
{
    VolumeData(const std::filesystem::path &path, std::size_t skipByteNum,
               const std::array<int, 3> &init_res, const Bounds3f &init_bound,
               int init_mipmapLevel, AlbedoVolume &&init_albedo)
        : resolution{ init_res }, bound{ init_bound },
          mipmapLevel{ init_mipmapLevel }
    {
        auto [density, elemNum] = ReadVoxels(path, skipByteNum, init_res);
        auto densityRawPtr = density.get();
        maxDensity = *std::max_element(densityRawPtr, densityRawPtr + elemNum);

        // Currently we only use the final layer in MIPMap.
        mipmap = MIPMap3D{ densityRawPtr, init_res[2], init_res[1], init_res[0],
                           init_mipmapLevel };

        if (init_albedo.impl_)
            albedo = std::move(*(init_albedo.impl_));
    }

    std::array<int, 3> resolution;
    Bounds3f bound;
    MIPMap3D mipmap;
    int mipmapLevel;
    float maxDensity;
    AlbedoVolume::Data albedo;
};

AlbedoVolume::AlbedoVolume(const std::filesystem::path &path,
                           std::size_t skipByteNum,
                           const std::array<int, 3> &init_res, int channelNum)
    : impl_{ std::make_unique<Data>(path, skipByteNum, init_res, channelNum) }
{
}

AlbedoVolume::AlbedoVolume() noexcept = default;
AlbedoVolume::AlbedoVolume(AlbedoVolume &&) noexcept = default;
AlbedoVolume &AlbedoVolume::operator=(AlbedoVolume &&) noexcept = default;
AlbedoVolume::~AlbedoVolume() = default;

GPUSimpleVolume::GPUSimpleVolume(const std::filesystem::path &path,
                                 std::size_t skipByteNum,
                                 const std::array<int, 3> &init_res,
                                 const Bounds3f &init_bound, int mipmapLevel,
                                 AlbedoVolume &&albedoVolume)
    : impl_{ std::make_unique<VolumeData>(path, skipByteNum, init_res,
                                          init_bound, mipmapLevel,
                                          std::move(albedoVolume)) }
{
}

GPUSimpleVolume::GPUSimpleVolume() noexcept = default;
GPUSimpleVolume::GPUSimpleVolume(GPUSimpleVolume &&) noexcept = default;
GPUSimpleVolume &GPUSimpleVolume::operator=(GPUSimpleVolume &&) noexcept =
    default;
GPUSimpleVolume::~GPUSimpleVolume() = default;

void GPUSimpleVolume::RecreateTexture(bool innerLinear, bool outerLinear)
{ // Compact mipmap level inside int, using signbit to denote cached or not.
    bool cached = impl_->mipmapLevel < 0;
    int level = cached ? ~impl_->mipmapLevel : impl_->mipmapLevel;

    int cachedLevel =
        impl_->mipmap.RecreateTexture(level, innerLinear, outerLinear, cached);
    impl_->mipmapLevel = ~cachedLevel;
}

VolumeKernelData::VolumeKernelData(const GPUSimpleVolume::VolumeData &v)
    : resolution_{ v.resolution[0], v.resolution[1], v.resolution[2] },
      bound_{ v.bound }, mipmap_{ v.mipmap },
      invMaxDensity_{ 1.0f / v.maxDensity }, albedoMipMap_{ v.albedo.mipmap }
{
}

__device__ float VolumeKernelData::GetDensityRaw_(
    glm::vec3 texCoord, int sampleLevel) const noexcept
{
    // make z orders first.
    return mipmap_.Sample(texCoord[2], texCoord[1], texCoord[0], sampleLevel);
}

__device__ float VolumeKernelData::GetDensity(glm::vec3 pos) const noexcept
{
    auto texCoord = bound_.Offset(pos); // + glm::vec3{ 0.5f } / resolution_;
    return GetDensityRaw_(texCoord);
}

__device__ glm::vec3 VolumeKernelData::GetAlbedo(glm::vec3 pos) const noexcept
{
    auto swizzledCoord = ToSampleCoord(pos);
    return GetAlbedoByCoord(swizzledCoord);
}

__device__ glm::vec3 VolumeKernelData::GetAlbedoByCoord(
    glm::vec3 coord) const noexcept
{
    if (!albedoMipMap_)
        return glm::vec3{ 1.0f }; // Plain albedo case.

    auto result = albedoMipMap_.Sample<float4>(coord[0], coord[1], coord[2], 0);
    return glm::vec3{ result.x, result.y, result.z };
}

__device__ float VolumeKernelData::GetNLogTr(glm::vec3 begin, glm::vec3 end,
                                             int sampleLevel) const noexcept
{
    float tMin, tMax;
    glm::vec3 rawDir = end - begin;
    if (!bound_.IntersectP(begin, rawDir, 1.0f, &tMin, &tMax))
        return 0.0f; // No intersection, no radiance loss.
    // ceil to make resolution actually integer, otherwise DDA marching
    // algorithm is wrong.
    auto resolution = glm::ceil(resolution_ / (1.0f * (1 << sampleLevel)));

    begin += tMin * rawDir, end -= (1.0f - tMax) * rawDir;

    // With such direction, tMax is fixed to be 1.0f.
    Ray rayGrid{ bound_.Offset(begin), (end - begin) / bound_.Diagonal() };
    float unitRayLength = glm::length(end - begin);
    if (unitRayLength < 1e-5f) // Two points are too close.
        return 0.0f;

    float nextCrossingT[3], deltaT[3];
    int step[3], voxelLimit[3], voxel[3];

    for (int axis = 0; axis < 3; ++axis)
    {
        // Initialize ray stepping parameters for _axis_
        // Compute current voxel for axis and handle negative zero direction
        voxel[axis] = glm::clamp<int>(rayGrid.ori[axis] * resolution[axis], 0,
                                      resolution[axis] - 1);
        deltaT[axis] = 1 / (abs(rayGrid.dir[axis]) * resolution[axis]);
        if (rayGrid.dir[axis] == -0.f)
            rayGrid.dir[axis] = 0.f;

        if (rayGrid.dir[axis] >= 0)
        {
            // Handle ray with positive direction for voxel stepping
            float nextVoxelPos = float(voxel[axis] + 1) / resolution[axis];
            nextCrossingT[axis] =
                (nextVoxelPos - rayGrid.ori[axis]) / rayGrid.dir[axis];
            step[axis] = 1;
            voxelLimit[axis] = resolution[axis];
        }
        else
        {
            // Handle ray with negative direction for voxel stepping
            float nextVoxelPos = float(voxel[axis]) / resolution[axis];
            nextCrossingT[axis] =
                (nextVoxelPos - rayGrid.ori[axis]) / rayGrid.dir[axis];
            step[axis] = -1;
            voxelLimit[axis] = -1;
        }
    }

    float lastT, currT = 0.0f;
    static constexpr float s_tMax = 1.0f;
    float lastDensity, thickness = 0.0f;

    bool endIteration = false;
    while (!endIteration)
    {
        int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
                   ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
                   ((nextCrossingT[1] < nextCrossingT[2]));
        constexpr int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
        int stepAxis = cmpToAxis[bits];
        float tVoxelExit = min(s_tMax, nextCrossingT[stepAxis]);

        // add 0.5f to prevent rounding error.
        lastDensity =
            max(0.f, GetDensityRaw_(glm::vec3{ voxel[0] + 0.5f, voxel[1] + 0.5f,
                                               voxel[2] + 0.5f } /
                                        resolution,
                                    sampleLevel));
        lastT = currT, currT = tVoxelExit;

        if (nextCrossingT[stepAxis] > s_tMax)
            endIteration = true, currT = s_tMax;
        voxel[stepAxis] += step[stepAxis];
        if (voxel[stepAxis] == voxelLimit[stepAxis])
            endIteration = true;
        nextCrossingT[stepAxis] += deltaT[stepAxis];
        thickness += lastDensity * (currT - lastT) * unitRayLength;
    }

    return thickness;
}

__device__ float VolumeKernelData::GetNLogTrMarching(
    glm::vec3 begin, glm::vec3 dir, glm::vec3 invDir, glm::ivec3 dirIsPos,
    int sampleLevel, int stepNum) const noexcept
{
    float t = bound_.IntersectUnchecked(begin, invDir, dirIsPos);
    assert(t >= 0);

    float delta = t / stepNum;
    float result = 0.0f;
    for (int i = 0; i < stepNum; i++)
    {
        begin += delta * dir;
        auto normalizedVoxel = bound_.Offset(begin);
        float density =
            max(0.f, GetDensityRaw_(normalizedVoxel + (0.5f / resolution_),
                                    sampleLevel));
        result += density;
    }

    // Well, length is normally 1.0f.
    result *= delta * glm::length(dir);
    return result;
}

__device__ float VolumeKernelData::GetEstimatedTrRatio(
    curandState *seed, glm::vec3 begin, glm::vec3 end) const noexcept
{
    float tr = 1;
    float t = 0;
    int loop_num = 0;

    glm::vec3 dir = end - begin;
    float maxDis = glm::length(dir);
    dir /= maxDis;

    float invMaxDensity = invMaxDensity_;
    // Ratio tracking.
    while (loop_num++ < 10000)
    {
        t -= log(1 - Rand(seed)) * invMaxDensity;

        if (t > maxDis)
            return tr;

        float density = GetDensity(begin + (dir * t));
        if (density <= 0)
        {
            t -= density;
            continue;
        }

        tr *= 1 - density * invMaxDensity;
    }
    return tr;
}

__device__ float VolumeKernelData::GetEstimatedTrDelta(
    curandState *seed, glm::vec3 begin, glm::vec3 end) const noexcept
{
    float tr = 1;
    float t = 0;
    int loop_num = 0;

    glm::vec3 dir = end - begin;
    float maxDis = glm::length(dir);
    dir /= maxDis;

    float invMaxDensity = invMaxDensity_;
    while (loop_num++ < 10000)
    {
        t -= log(1 - Rand(seed)) * invMaxDensity;

        if (t > maxDis)
            break;

        float density = GetDensity(begin + (dir * t));
        if (density <= 0)
        {
            t -= density;
            continue;
        }

        if (Rand(seed) < density * invMaxDensity)
        {
            tr = 0;
            break;
        }
    }
    return tr;
}

__device__ glm::vec4 VolumeKernelData::GetSamplePoint(
    curandState *seed, glm::vec3 begin, glm::vec3 dir) const noexcept
{
    glm::vec4 impossiblePoint{ begin + dir, -1.0f };
    float tMin, tMax;
    if (!bound_.IntersectP(begin, dir, INFINITY, &tMin, &tMax))
        return impossiblePoint;

    begin += tMin * dir;
    // Then do delta tracking
    float t = 0, maxDis = tMax - tMin;
    int loop_num = 0;
    float invMaxDensity = invMaxDensity_;

    while (loop_num++ < 10000)
    {
        t -= log(1 - Rand(seed)) * invMaxDensity;

        if (t > maxDis)
            break;

        auto newPos = begin + (dir * t);
        float density = GetDensity(newPos);
        if (density <= 0)
        {
            t -= density;
            continue;
        }

        if (Rand(seed) < density * invMaxDensity)
            return glm::vec4{ newPos, maxDis - t };
    }
    return impossiblePoint;
}

// ----------------- TransmittanceMap ---------------------
struct TransmittanceMap::MapData
{
    UniqueResource<float *, CUDADeleter> buffer;
    MutableMIPMap3D map;
    glm::vec3 lastDir{ 0 };

    MapData(UniqueResource<float *, CUDADeleter> &&init_buffer,
            MutableMIPMap3D &&init_map)
        : buffer{ std::move(init_buffer) }, map{ std::move(init_map) }
    {
    }
};

TrMapKernelData::TrMapKernelData(const TransmittanceMap::MapData &mapData)
    : map{ mapData.map }
{
}

static __global__ void GenerateTransmittanceMap(VolumeKernelData data, int resX,
                                                int resY, int resZ,
                                                int sampleLevel, glm::vec3 dir,
                                                float *dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resX || y >= resY || z >= resZ)
        return;

    auto idx = x * resY * resZ + y * resZ + z;

    auto bound = data.GetBound();
    glm::vec3 fract{ (x + 0.5f) / resX, (y + 0.5f) / resY, (z + 0.5f) / resZ };
    auto gridPos = bound.Min() + fract * bound.Diagonal();
    auto gridEndPos = gridPos + glm::length(bound.Diagonal()) * dir;
    dst[idx] = data.GetNLogTr(gridPos, gridEndPos, sampleLevel);
    assert(dst[idx] >= 0);
    return;
}

static __global__ void GenerateTransmittanceMapByMarching(
    VolumeKernelData data, int resX, int resY, int resZ, int sampleLevel,
    int stepNum, glm::vec3 dir, glm::vec3 invDir, glm::ivec3 dirIsPos,
    float *dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resX || y >= resY || z >= resZ)
        return;

    auto idx = x * resY * resZ + y * resZ + z;

    auto bound = data.GetBound();
    glm::vec3 fract{ (x + 0.5f) / resX, (y + 0.5f) / resY, (z + 0.5f) / resZ };
    auto gridPos = bound.Min() + fract * bound.Diagonal();
    dst[idx] = data.GetNLogTrMarching(gridPos, dir, invDir, dirIsPos,
                                      sampleLevel, stepNum);
    assert(dst[idx] >= 0);
    return;
}

TransmittanceMap::TransmittanceMap() noexcept = default;
TransmittanceMap::TransmittanceMap(TransmittanceMap &&) noexcept = default;
TransmittanceMap &TransmittanceMap::operator=(TransmittanceMap &&) noexcept =
    default;
TransmittanceMap::~TransmittanceMap() = default;

TransmittanceMap::TransmittanceMap(
    const GPUSimpleVolume::VolumeData &volumeData, int beginLevel,
    int mipmapLevel)
    : TransmittanceMap{ { volumeData.resolution[0] >> beginLevel,
                          volumeData.resolution[1] >> beginLevel,
                          volumeData.resolution[2] >> beginLevel },
                        mipmapLevel }
{
}

TransmittanceMap::TransmittanceMap(std::array<int, 3> res, int mipmapLevel)
{
    auto elemNum = static_cast<std::size_t>(res[0]) * res[1] * res[2];
    UniqueResource<float *, CUDADeleter> baseBuffer;
    float *transmittanceField;
    CheckCUDAError(cudaMalloc(&transmittanceField, elemNum * sizeof(float)));
    baseBuffer.Reloc(transmittanceField);

    impl_ = std::make_unique<MapData>(
        std::move(baseBuffer),
        MutableMIPMap3D{ res[0], res[1], res[2], mipmapLevel, true });
}

void TransmittanceMap::Update(const GPUSimpleVolume::VolumeData &volumeData,
                              glm::vec3 lightDir, int sampleLevel, int stepNum,
                              bool forced) const
{
    assert(impl_);
    if (lightDir == impl_->lastDir && !forced)
        return;

    impl_->lastDir = lightDir;
    auto res = impl_->map.GetResolution();
    auto buffer = impl_->buffer.Get();

    dim3 block(8, 8, 4);
    dim3 grid(RoundUpDiv((unsigned int)res.width, block.x),
              RoundUpDiv((unsigned int)res.height, block.y),
              RoundUpDiv((unsigned int)res.depth, block.z));

    if (stepNum == 0)
    {
        GenerateTransmittanceMap<<<grid, block>>>(
            VolumeKernelData{ volumeData }, res.width, res.height, res.depth,
            sampleLevel, lightDir, buffer);
    }
    else
    {
        GenerateTransmittanceMapByMarching<<<grid, block>>>(
            VolumeKernelData{ volumeData }, res.width, res.height, res.depth,
            sampleLevel, stepNum, lightDir, 1.0f / lightDir,
            glm::ivec3{ glm::greaterThanEqual(lightDir, glm::vec3{ 0 }) },
            buffer);
    }
    CheckCUDAError(cudaGetLastError());

    impl_->map.Update(buffer);
    impl_->map.Sync();
}