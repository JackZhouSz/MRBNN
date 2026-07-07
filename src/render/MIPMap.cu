/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "MIPMap.hpp"
#include <bit>
#include <iostream>
#include <vector>

static std::pair<dim3, dim3> GetDispatchDims(cudaExtent extent)
{
    dim3 block(8, 8, 4);
    dim3 grid(RoundUpDiv(extent.width, block.x),
              RoundUpDiv(extent.height, block.y),
              RoundUpDiv(extent.depth, block.z));
    return { block, grid };
}

static cudaSurfaceObject_t CreateSurf(cudaArray_t arr)
{
    cudaSurfaceObject_t surfObj;
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;
    CheckCUDAError(cudaCreateSurfaceObject(&surfObj, &resDesc));
    return surfObj;
}

static cudaArray_t CreateArray(cudaExtent ex)
{
    cudaArray_t arr{};
    auto desc = cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0,
                                      cudaChannelFormatKindFloat);

    CheckCUDAError(
        cudaMalloc3DArray(&arr, &desc, ex, cudaArraySurfaceLoadStore));
    return arr;
}

static bool ArrayExtentMatched(cudaArray_t a, cudaArray_t b)
{
    cudaExtent ex1, ex2;
    CheckCUDAError(cudaArrayGetInfo(nullptr, &ex1, nullptr, a));
    CheckCUDAError(cudaArrayGetInfo(nullptr, &ex2, nullptr, b));

    bool success = ex1.width == ex2.width && ex1.height == ex2.height &&
                   ex1.depth == ex2.depth;

    if (!success)
    {
        std::cout << "Unmatched array: (" << ex1.width << ", " << ex1.height
                  << ", " << ex1.depth << ") <-> (" << ex2.width << ", "
                  << ex2.height << ", " << ex2.depth << ")\n";
    }

    return success;
}

static cudaExtent GetNextExtent(cudaExtent ex)
{
    ex.width = std::max(ex.width / 2, std::size_t{ 1 });
    ex.height = std::max(ex.height / 2, std::size_t{ 1 });
    ex.depth = std::max(ex.depth / 2, std::size_t{ 1 });
    return ex;
}

static cudaExtent ExchangeArray(cudaExtent ex, auto &vecs, auto &lastSurf,
                                auto &currSurf, auto &lastArr, auto &currArr)
{
    ex = GetNextExtent(ex);
    vecs.emplace_back(CreateArray(ex));

    lastSurf = std::move(currSurf);
    lastArr = currArr;
    currArr = vecs.back().Get();
    currSurf.Reloc(CreateSurf(currArr));

    return ex;
}

// Very strangely, if we pass (..., cudaExtent srcExtent), then
// it will be modified during copying from host to kernel ?!?!?!?!
// (3,3,3) -> (0,3,0) in our unit test.
static __global__ void GenerateMIPMap(cudaSurfaceObject_t src,
                                      cudaSurfaceObject_t dst, int srcW,
                                      int srcH, int srcD)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x >= srcW / 2 && x != 0) || (y >= srcH / 2 && y != 0) ||
        (z >= srcD / 2 && z != 0))
        return;

    int xi = x * 2, yi = y * 2, zi = z * 2;
    auto co = [&] __device__(int dx, int dy, int dz) {
        return surf3Dread<float>(src, (xi + dx) * sizeof(float), yi + dy,
                                 zi + dz, cudaBoundaryModeClamp);
    };

    bool hasTwoX = (xi + 1) < srcW;
    bool hasTwoY = (yi + 1) < srcH;
    bool hasTwoZ = (zi + 1) < srcD;

    float value = 0.f;
    // Likely to be this case.
    if (hasTwoX && hasTwoY && hasTwoZ)
    {
        value = (co(0, 0, 0) + co(1, 0, 0) + co(0, 1, 0) + co(1, 1, 0) +
                 co(0, 0, 1) + co(1, 0, 1) + co(0, 1, 1) + co(1, 1, 1)) /
                8;
    }
    else
    {
        int totalCnt = 0;
        for (int dx = 0; dx <= (hasTwoX ? 1 : 0); dx++)
            for (int dy = 0; dy <= (hasTwoY ? 1 : 0); dy++)
                for (int dz = 0; dz <= (hasTwoZ ? 1 : 0); dz++)
                {
                    value += co(dx, dy, dz);
                    totalCnt++;
                }
        value /= totalCnt;
    }

    surf3Dwrite(value, dst, x * sizeof(float), y, z, cudaBoundaryModeClamp);
}

static cudaTextureObject_t CreateTextureFromMipmappedArray(
    int mipmapLevel, cudaMipmappedArray_t mipmapRawArr, bool innerLinear,
    bool outerLinear)
{
    cudaTextureDesc desc{};
    desc.addressMode[0] = cudaAddressModeClamp;
    desc.addressMode[1] = cudaAddressModeClamp;
    desc.addressMode[2] = cudaAddressModeClamp;

    desc.filterMode = innerLinear ? cudaFilterModeLinear : cudaFilterModePoint;
    desc.readMode = cudaReadModeElementType;

    desc.sRGB = 0; // Linear colors in the HDR image.
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
    // shows that mipmap MUST have a normalized coordinate.
    desc.normalizedCoords = 1;
    desc.maxAnisotropy = 1;

    desc.mipmapFilterMode =
        outerLinear ? cudaFilterModeLinear : cudaFilterModePoint;
    desc.mipmapLevelBias = 0.0f;
    desc.minMipmapLevelClamp = 0.0f;
    desc.maxMipmapLevelClamp = mipmapLevel;

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = mipmapRawArr;

    cudaTextureObject_t temp;
    CheckCUDAError(cudaCreateTextureObject(&temp, &resDesc, &desc, nullptr));
    return temp;
}

static void CopyToArray(float *baseVoxels, cudaExtent extent,
                        cudaArray_t dstArr, int channelNum = 1)
{
    cudaMemcpy3DParms params{};
    params.srcPtr = make_cudaPitchedPtr(
        baseVoxels,                                // Host data pointer
        extent.width * channelNum * sizeof(float), // Pitch (bytes per row)
        extent.width,                              // Logical width (elements)
        extent.height                              // Logical height (rows)
    );
    params.dstArray = dstArr;
    params.kind = cudaMemcpyDefault;
    params.extent = extent;
    CheckCUDAError(cudaMemcpy3D(&params));
}

static void CopyToArray(cudaArray_t srcArr, cudaExtent extent,
                        cudaArray_t dstArr)
{
    cudaMemcpy3DParms params{};
    params.srcArray = srcArr;
    params.dstArray = dstArr;
    params.kind = cudaMemcpyDeviceToDevice;
    params.extent = extent;
    CheckCUDAError(cudaMemcpy3D(&params));
}

static void CopyToMipmap(cudaArray_t srcArr, cudaExtent extent,
                         cudaMipmappedArray_t mipmapArr, int idx)
{
    cudaArray_t dstArr;
    CheckCUDAError(cudaGetMipmappedArrayLevel(&dstArr, mipmapArr, idx));
    assert(ArrayExtentMatched(srcArr, dstArr));

    cudaMemcpy3DParms params{};
    params.srcArray = srcArr;
    params.dstArray = dstArr;
    params.kind = cudaMemcpyDeviceToDevice;
    params.extent = extent;
    CheckCUDAError(cudaMemcpy3D(&params));
}

static void CopyToMipmap(float *baseVoxels, cudaExtent extent,
                         cudaMipmappedArray_t mipmapArr, int idx,
                         int channelNum = 1)
{
    cudaArray_t dstArr;
    CheckCUDAError(cudaGetMipmappedArrayLevel(&dstArr, mipmapArr, idx));

    cudaMemcpy3DParms params{};
    params.srcPtr = make_cudaPitchedPtr(
        baseVoxels,                                // Host data pointer
        extent.width * channelNum * sizeof(float), // Pitch (bytes per row)
        extent.width,                              // Logical width (elements)
        extent.height                              // Logical height (rows)
    );
    params.dstArray = dstArr;
    params.kind = cudaMemcpyDefault;
    params.extent = extent;
    CheckCUDAError(cudaMemcpy3D(&params));
}

MIPMap3D::MIPMap3D(float *baseVoxels, int resX, int resY, int resZ,
                   int mipmapLevel, bool innerLinear, bool outerLinear,
                   int channelNum)
{
    cudaChannelFormatDesc channelDesc;
    if (channelNum == 1)
        channelDesc = cudaCreateChannelDesc<float>();
    else if (channelNum == 2)
        channelDesc = cudaCreateChannelDesc<float2>();
    else if (channelNum == 4)
        channelDesc = cudaCreateChannelDesc<float4>();
    else
    {
        throw std::invalid_argument{
            "Only support 1/2/4 channel(s) for texture."
        };
    }

    if (channelNum > 1 && mipmapLevel > 0)
    {
        throw std::invalid_argument{
            "Currently channels more than 1 doesn't support mipmap."
        };
    }

    cudaMipmappedArray_t mipmapRawArr{};
    cudaExtent extent(resX, resY, resZ);
    CheckCUDAError(cudaMallocMipmappedArray(&mipmapRawArr, &channelDesc, extent,
                                            mipmapLevel + 1));
    mipmap_.Reloc(mipmapRawArr);

    if (mipmapLevel == 0 || baseVoxels == nullptr)
    {
        if (baseVoxels != nullptr)
            CopyToMipmap(baseVoxels, extent, mipmapRawArr, 0, channelNum);
        tex_.Reloc(CreateTextureFromMipmappedArray(mipmapLevel, mipmapRawArr,
                                                   innerLinear, outerLinear));
        return;
    }

    std::vector<UniqueResource<cudaArray_t, CUDAArrayDeleter>> vecs;
    vecs.reserve(mipmapLevel + 1);
    auto baseArr = vecs.emplace_back(CreateArray(extent)).Get();
    CopyToArray(baseVoxels, extent, baseArr);
    CopyToMipmap(baseArr, extent, mipmapRawArr, 0);

    UniqueResource<cudaSurfaceObject_t, CUDASurfaceDeleter> lastSurf,
        currSurf{ CreateSurf(baseArr) };
    cudaArray_t lastArr, currArr = baseArr;
    int currLevel = 1;
    for (cudaExtent lastExtent = extent;
         currLevel <= mipmapLevel &&
         (lastExtent.width > 1 || lastExtent.height > 1 ||
          lastExtent.depth > 1);
         currLevel++)
    {
        auto newExtent = ExchangeArray(lastExtent, vecs, lastSurf, currSurf,
                                       lastArr, currArr);

        auto [block, grid] = GetDispatchDims(newExtent);
        GenerateMIPMap<<<grid, block>>>(lastSurf.Get(), currSurf.Get(),
                                        lastExtent.width, lastExtent.height,
                                        lastExtent.depth);
        CheckCUDAError(cudaGetLastError());
        cudaDeviceSynchronize();
        CopyToMipmap(currArr, newExtent, mipmapRawArr, currLevel);
        lastExtent = newExtent;
    }
    tex_.Reloc(CreateTextureFromMipmappedArray(currLevel - 1, mipmapRawArr,
                                               innerLinear, outerLinear));
}

int MIPMap3D::RecreateTexture(int expectedLevel, bool innerLinear,
                              bool outerLinear, bool cachedLevel)
{
    if (!cachedLevel)
    {
        cudaArray_t dstArr;
        CheckCUDAError(cudaGetMipmappedArrayLevel(&dstArr, mipmap_.Get(), 0));
        cudaExtent extent;
        CheckCUDAError(cudaArrayGetInfo(nullptr, &extent, nullptr, dstArr));
        auto maxExtent =
            std::max({ extent.width, extent.height, extent.depth });
        auto maxLevel = std::bit_width(maxExtent);
        expectedLevel = std::min(maxLevel, expectedLevel);
    }

    tex_.Reset(CreateTextureFromMipmappedArray(expectedLevel, mipmap_.Get(),
                                               innerLinear, outerLinear));
    return expectedLevel;
}

MutableMIPMap3D::Cache::Cache(cudaArray_t mem, cudaArray_t memInMIPMap)
    : mem_{ mem }, surf_{ CreateSurf(mem) }, memInMIPMap_{ memInMIPMap }
{
}

MutableMIPMap3D::MutableMIPMap3D(int resX, int resY, int resZ, int mipmapLevel,
                                 bool innerLinear, bool outerLinear)
    : MIPMap3D{ nullptr, resX, resY, resZ, mipmapLevel, innerLinear, outerLinear },
      resolution_{ make_cudaExtent(resX, resY, resZ) }
{
    auto mipmapArr = mipmap_.Get();
    cudaArray_t dstArr;
    CheckCUDAError(cudaGetMipmappedArrayLevel(&dstArr, mipmapArr, 0));
    updateCache_.emplace_back(CreateArray(resolution_), dstArr);
    int currLevel = 1;
    for (cudaExtent lastExtent = resolution_;
         currLevel <= mipmapLevel &&
         (lastExtent.width > 1 || lastExtent.height > 1 ||
          lastExtent.depth > 1);
         currLevel++)
    {
        auto newExtent = GetNextExtent(lastExtent);
        auto newArr = CreateArray(newExtent);

        CheckCUDAError(
            cudaGetMipmappedArrayLevel(&dstArr, mipmapArr, currLevel));
        assert(ArrayExtentMatched(newArr, dstArr));

        updateCache_.emplace_back(newArr, dstArr);
        lastExtent = newExtent;
    }
}

void MutableMIPMap3D::Update(float *baseVoxels) const
{
    if (baseVoxels == nullptr || updateCache_.empty())
        return;

    // Copy to the first surface.
    CopyToArray(baseVoxels, resolution_, updateCache_[0].GetSrcMemory());
    auto extent = resolution_;
    for (std::size_t i = 1; i < updateCache_.size(); i++)
    {
        auto newExtent = GetNextExtent(extent);
        auto [block, grid] = GetDispatchDims(newExtent);
        GenerateMIPMap<<<grid, block>>>(
            updateCache_[i - 1].GetSurface(), updateCache_[i].GetSurface(),
            extent.width, extent.height, extent.depth);
        CheckCUDAError(cudaGetLastError());
        extent = newExtent;
    }
    return;
}

void MutableMIPMap3D::Sync() const
{
    // This sync seems unnecessary as we only use default stream.
    auto extent = resolution_;
    for (const auto &cache : updateCache_)
    {
        CopyToArray(cache.GetSrcMemory(), extent, cache.GetDstMemory());
        extent = GetNextExtent(extent);
    }
}