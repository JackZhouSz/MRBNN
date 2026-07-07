/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "Texture2D.hpp"
#include "stbi/stb_image.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include <execution>
#include <numeric>
#include <type_traits>

using namespace std::literals;

namespace
{

template<typename T>
class STBDeleter
{
public:
    void operator()(T *ptr) const { stbi_image_free(ptr); }
};

template<typename T>
using STBUniquePtr = std::unique_ptr<T, STBDeleter<T>>;

template<typename T>
static inline STBUniquePtr<T> STBLoadImage(const std::filesystem::path &path,
                                           int &width, int &height,
                                           int &channels)
{
    std::ifstream fin{ path, std::ios::binary };
    fin.exceptions(std::ios::failbit | std::ios::badbit);

    std::ostringstream image;
    image << fin.rdbuf();
    auto memBuffer = image.str();
    auto imagePtr = reinterpret_cast<const stbi_uc *>(memBuffer.c_str());
    int size;
    {
        auto rawSize = memBuffer.size() / sizeof(stbi_uc);
        assert(memBuffer.size() % sizeof(stbi_uc) == 0 &&
               rawSize <= std::numeric_limits<int>::max());
        size = static_cast<int>(rawSize);
        CheckError(stbi_info_from_memory(imagePtr, size, &width, &height,
                                         &channels) != 0,
                   "Unable to load image.");
    }

    int expectChannels = 0;
    if (channels == 3)
    {
        std::cerr << "CUDA texture only supports 1, 2, 4 channels; image " +
                         path.string() + " rgb will be turned into rgba.\n";
        channels = expectChannels = 4;
    }

    int d1, d2, d3; // discarded since we've already got them from stbi_info.
    STBUniquePtr<T> ptr;
    if constexpr (std::is_same_v<T, unsigned char>)
        ptr = STBUniquePtr<T>{ stbi_load_from_memory(imagePtr, size, &d1, &d2,
                                                     &d3, expectChannels) };
    else if constexpr (std::is_same_v<T, unsigned short>)
        ptr = STBUniquePtr<T>{ stbi_load_16_from_memory(
            imagePtr, size, &d1, &d2, &d3, expectChannels) };
    else if constexpr (std::is_same_v<T, float>)
        ptr = STBUniquePtr<T>{ stbi_loadf_from_memory(imagePtr, size, &d1, &d2,
                                                      &d3, expectChannels) };

    CheckError(ptr != nullptr, LAZY_STR("Cannot load image " + path.string() +
                                        ", reason: "s + stbi_failure_reason()));

    return ptr;
}

template<typename T>
static inline cudaChannelFormatDesc CreateChannelFormatDesc(int channelNum)
{
    constexpr auto size = sizeof(T) * 8;
    constexpr cudaChannelFormatKind kind = std::is_integral_v<T>
                                               ? cudaChannelFormatKindUnsigned
                                               : cudaChannelFormatKindFloat;
    switch (channelNum)
    {
    case 1:
        return cudaCreateChannelDesc(size, 0, 0, 0, kind);
    case 2:
        return cudaCreateChannelDesc(size, size, 0, 0, kind);
    case 4:
        return cudaCreateChannelDesc(size, size, size, size, kind);
    default:
        throw std::runtime_error{ "Unrecognized texture format." };
    }
    return cudaChannelFormatDesc{};
}

template<typename T>
std::unique_ptr<T[]> TransferChannels3To4(T *src, int width, int height)
{
    auto size = static_cast<std::size_t>(width) * height;
    auto result = std::make_unique_for_overwrite<T[]>(size * 4);
    for (std::size_t i = 0; i < size; i++)
    {
        for (int j = 0; j < 3; j++)
            result[i * 4 + j] = src[i * 3 + j];
        result[i * 4 + 3] = T{ 0 };
    }
    return result;
}

} // namespace

template<typename T>
void Texture2D::LoadResources_(T *ptr, int width, int height, int channels)
{
    std::unique_ptr<T[]> possibleBuffer;
    if (channels == 3)
    {
        possibleBuffer = TransferChannels3To4(ptr, width, height);
        ptr = possibleBuffer.get();
        channels = 4;
    }

    auto channelDesc = CreateChannelFormatDesc<T>(channels);

    buffer_.Reloc([&]() {
        cudaArray_t buffer;
        CheckCUDAError(cudaMallocArray(&buffer, &channelDesc, width, height));
        return buffer;
    }());

    auto bytesPerElem = channels * sizeof(T);
    // stb_image doesn't have any padding, so width == spitch.
    CheckCUDAError(cudaMemcpy2DToArray(
        buffer_.Get(), 0, 0, ptr, width * bytesPerElem, width * bytesPerElem,
        height, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = buffer_.Get();

    cudaTextureDesc texDesc{};
    // NOTICE: currently, we do not let users to specify wrap mode themselves.
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // NOTICE: currently, we don't specify cudaResourceViewDesc.
    tex_.Reloc([&]() {
        cudaTextureObject_t tex;
        CheckCUDAError(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
        return tex;
    }());
}

template<typename T>
void Texture2D::LoadResources_(const std::filesystem::path &path)
{
    int width, height, channels;
    auto ptr = STBLoadImage<T>(path, width, height, channels);
    LoadResources_<T>(ptr.get(), width, height, channels);
}

Texture2D::Texture2D(const std::filesystem::path &path, Format format)
{
    switch (format)
    {
    case Format::UByte:
        LoadResources_<unsigned char>(path);
        return;
    case Format::UShort:
        LoadResources_<unsigned short>(path);
        return;
    case Format::Float:
        LoadResources_<float>(path);
        return;
    default:
        throw std::runtime_error{ "Unrecognized texture format." };
    }
    return;
}

Texture2D::Texture2D(void *ptr, int width, int height, int channelNum,
                     Format format)
{
    switch (format)
    {
    case Format::UByte:
        LoadResources_(static_cast<unsigned char *>(ptr), width, height,
                       channelNum);
        return;
    case Format::UShort:
        LoadResources_(static_cast<unsigned short *>(ptr), width, height,
                       channelNum);
        return;
    case Format::Float:
        LoadResources_(static_cast<float *>(ptr), width, height, channelNum);
        return;
    default:
        throw std::runtime_error{ "Unrecognized texture format." };
    }
    return;
}

namespace
{
cudaSurfaceObject_t CreateSurf(cudaArray_t arr)
{
    cudaSurfaceObject_t surfObj;
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;
    CheckCUDAError(cudaCreateSurfaceObject(&surfObj, &resDesc));
    return surfObj;
}

cudaArray_t CreateArray(cudaChannelFormatDesc desc, int width, int height)
{
    cudaArray_t buffer;
    CheckCUDAError(cudaMallocArray(&buffer, &desc, width, height));
    return buffer;
}

bool ArrayExtentMatched(cudaArray_t a, cudaArray_t b)
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

template<typename T, typename U>
cudaExtent ExchangeArray(cudaChannelFormatDesc desc, cudaExtent ex, T &vecs,
                         U &lastSurf, U &currSurf, cudaArray_t &lastArr,
                         cudaArray_t &currArr)
{
    ex.width = std::max(ex.width / 2, 1ull);
    ex.height = std::max(ex.height / 2, 1ull);

    vecs.emplace_back(CreateArray(desc, ex.width, ex.height));

    lastSurf = std::move(currSurf);
    lastArr = currArr;
    currArr = vecs.back().Get();
    currSurf.Reloc(CreateSurf(currArr));

    return ex;
}

template<typename ElemType, int N>
struct ElementHelper
{
};

template<>
struct ElementHelper<float, 4>
{
    using CalType = glm::vec4;
    using SampleType = float4;
    __device__ static CalType ToCalType(SampleType s)
    {
        return { s.x, s.y, s.z, s.w };
    }
    __device__ static SampleType ToSampleType(CalType c)
    {
        return { c[0], c[1], c[2], c[3] };
    }
};

template<typename ElemType, int N>
__global__ void GenerateMIPMap(cudaSurfaceObject_t src, cudaSurfaceObject_t dst,
                               int srcW, int srcH)
{
    using Helper = ElementHelper<ElemType, N>;
    using SampleType = typename Helper::SampleType;
    using CalType = typename Helper::CalType;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= srcW / 2 && x != 0) || (y >= srcH / 2 && y != 0))
        return;

    int xi = x * 2, yi = y * 2;
    auto co = [&] __device__(int dx, int dy) {
        return Helper::ToCalType(
            surf2Dread<SampleType>(src, (xi + dx) * sizeof(SampleType), yi + dy,
                                   cudaBoundaryModeClamp));
    };

    bool hasTwoX = (xi + 1) < srcW;
    bool hasTwoY = (yi + 1) < srcH;

    CalType value{};
    if (hasTwoX && hasTwoY)
    {
        value = (co(0, 0) + co(1, 0) + co(0, 1) + co(1, 1)) / ElemType{ 4 };
    }
    else
    {
        int totalCnt = 0;
        for (int dx = 0; dx <= (hasTwoX ? 1 : 0); dx++)
            for (int dy = 0; dy <= (hasTwoY ? 1 : 0); dy++)
            {
                value += co(dx, dy);
                totalCnt++;
            }
        value /= totalCnt;
    }

    surf2Dwrite(Helper::ToSampleType(value), dst, x * sizeof(SampleType), y,
                cudaBoundaryModeClamp);
}

template<typename T>
void CopyToArray(T *hostPtr, int width, int height, int channels,
                 cudaArray_t dstArr)
{
    auto bytesPerElem = channels * sizeof(T);
    // stb_image doesn't have any padding, so width == spitch.
    CheckCUDAError(cudaMemcpy2DToArray(
        dstArr, 0, 0, hostPtr, width * bytesPerElem, width * bytesPerElem,
        height, cudaMemcpyHostToDevice));
}

template<typename T>
void CopyToMipmap(cudaArray_t srcArr, int width, int height, int channels,
                  cudaMipmappedArray_t mipmapArr, int idx)
{
    auto bytesPerElem = channels * sizeof(T);

    cudaArray_t dstArr;
    CheckCUDAError(cudaGetMipmappedArrayLevel(&dstArr, mipmapArr, idx));
    assert(ArrayExtentMatched(srcArr, dstArr));
    CheckCUDAError(cudaMemcpy2DArrayToArray(dstArr, 0, 0, srcArr, 0, 0,
                                            width * bytesPerElem, height));
}

template<typename T>
void CopyToMipmap(T *hostPtr, int width, int height, int channels,
                  cudaMipmappedArray_t mipmapArr, int idx)
{
    cudaArray_t dstArr;
    CheckCUDAError(cudaGetMipmappedArrayLevel(&dstArr, mipmapArr, idx));
    CopyToArray(hostPtr, width, height, channels, dstArr);
}

cudaTextureObject_t CreateTextureFromMipmappedArray(
    int mipmapLevel, cudaMipmappedArray_t mipmapRawArr)
{
    cudaTextureDesc desc{};
    desc.addressMode[0] = cudaAddressModeWrap;
    desc.addressMode[1] = cudaAddressModeWrap;

    desc.filterMode = cudaFilterModeLinear;
    desc.readMode = cudaReadModeElementType;

    desc.sRGB = 0;
    desc.normalizedCoords = 1;
    desc.maxAnisotropy = 1;

    desc.mipmapFilterMode = cudaFilterModeLinear;
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

} // namespace

template<typename T>
void MIPMap2D::LoadResources_(const std::filesystem::path &path,
                              int mipmapLevel)
{
    mipmapLevel = std::min(mipmapLevel, 64);

    int width, height, channels;
    auto ptr = STBLoadImage<T>(path, width, height, channels);
    // Use double to avoid precision problem.
    std::size_t elemNum = (std::size_t)width * height * channels;
    estimatedAveragePower_ =
        static_cast<float>(std::reduce(std::execution::par_unseq, ptr.get(),
                                       ptr.get() + elemNum, 0.0) /
                           elemNum);
    auto channelDesc = CreateChannelFormatDesc<T>(channels);

    cudaMipmappedArray_t mipmapRawArr{};
    cudaExtent extent = make_cudaExtent(width, height, 0);
    CheckCUDAError(cudaMallocMipmappedArray(&mipmapRawArr, &channelDesc, extent,
                                            mipmapLevel + 1));
    mipmap_.Reloc(mipmapRawArr);

    if (mipmapLevel == 0)
    {
        CopyToMipmap(ptr.get(), width, height, channels, mipmapRawArr, 0);
        tex_.Reloc(CreateTextureFromMipmappedArray(mipmapLevel, mipmapRawArr));
        return;
    }

    std::vector<UniqueResource<cudaArray_t, CUDAArrayDeleter>> vecs;
    vecs.reserve(mipmapLevel + 1);
    auto baseArr =
        vecs.emplace_back(CreateArray(channelDesc, width, height)).Get();
    CopyToArray(ptr.get(), extent.width, extent.height, channels, baseArr);
    CopyToMipmap<T>(baseArr, width, height, channels, mipmapRawArr, 0);

    UniqueResource<cudaSurfaceObject_t, CUDASurfaceDeleter> lastSurf,
        currSurf{ CreateSurf(baseArr) };
    cudaArray_t lastArr, currArr = baseArr;
    int currLevel = 1;
    for (cudaExtent lastExtent = extent;
         currLevel <= mipmapLevel &&
         (lastExtent.width > 1 || lastExtent.height > 1);
         currLevel++)
    {
        auto newExtent = ExchangeArray(channelDesc, lastExtent, vecs, lastSurf,
                                       currSurf, lastArr, currArr);

        dim3 block(8, 4);
        dim3 grid(RoundUpDiv(newExtent.width, block.x),
                  RoundUpDiv(newExtent.height, block.y));
        // We haven't written more cases here.
        assert(channels == 4 && (std::is_same_v<T, float>));
        GenerateMIPMap<float, 4>
            <<<grid, block>>>(lastSurf.Get(), currSurf.Get(), lastExtent.width,
                              lastExtent.height);
        CheckCUDAError(cudaGetLastError());
        cudaDeviceSynchronize();
        CopyToMipmap<T>(currArr, newExtent.width, newExtent.height, channels,
                        mipmapRawArr, currLevel);
        lastExtent = newExtent;
    }
    tex_.Reloc(CreateTextureFromMipmappedArray(currLevel - 1, mipmapRawArr));
}

MIPMap2D::MIPMap2D(const std::filesystem::path &path, int mipmapLevel,
                   Format format)
{
    switch (format)
    {
    case Format::UByte:
        LoadResources_<unsigned char>(path, mipmapLevel);
        return;
    case Format::UShort:
        LoadResources_<unsigned short>(path, mipmapLevel);
        return;
    case Format::Float:
        LoadResources_<float>(path, mipmapLevel);
        return;
    default:
        throw std::runtime_error{ "Unrecognized texture format." };
    }
    return;
}
