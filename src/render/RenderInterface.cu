/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "EnvironmentBaking.hpp"
#include "RenderPayload.hpp"
#include "ToneMapping.cuh"

#define DLL_MACRO_NEED_IMPORT
#include "ExternalTCNN.hpp"

#include <iostream>
#include <numbers>

#define FEATURE_DIM_UPPER_BOUND 48
#define STEP_UPPER_BOUND 8
#define LIGHT_DIM_UPPER_BOUND 8

#ifdef DEBUG
#define CHECK_COORD(vec)                                           \
    if (!glm::all(glm::greaterThanEqual(vec, decltype(vec){ 0 }))) \
    {                                                              \
        printf("Out of minimum bound: ");                          \
        for (int i = 0; i < vec.length(); i++)                     \
            printf("%f ", vec[i]);                                 \
        assert(false);                                             \
    }                                                              \
    if (!glm::all(glm::lessThan(vec, decltype(vec){ 1.001 })))     \
    {                                                              \
        printf("Out of maxmimum bound: ");                         \
        for (int i = 0; i < vec.length(); i++)                     \
            printf("%f ", vec[i]);                                 \
        assert(false);                                             \
    }
#else
#define CHECK_COORD(vec)
#endif

namespace
{
static constexpr float pi = std::numbers::pi_v<float>;
static constexpr float inv_pi = std::numbers::inv_pi_v<float>;
static constexpr float inv_2pi = inv_pi / 2.0f;
static constexpr float inv_4pi = inv_pi / 4.0f;
static constexpr float e = std::numbers::e_v<float>;

__device__ int Hash(int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__device__ void InitRand(curandState *seed, int offset)
{
    int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
    curand_init(offset + Hash(i), 0, 0, seed);
}

__device__ float HenyeyGreenstein(float cosTheta, float g)
{
    float gSquare = g * g;
    float numerator = 1 - gSquare;
    float invDenominator = pow(1 + gSquare - 2 * g * cosTheta, -1.5);
    float phase = numerator * invDenominator * inv_4pi;
    return phase;
}

__device__ glm::vec2 ToNormalizedSphericalCoord(glm::vec3 dir)
{
    float theta = acos(glm::clamp(dir.z, -1.0f, 1.0f));
    float theta_norm = theta * inv_pi;

    float phi = atan2(dir.y, dir.x);
    phi = (phi < 0) ? (phi + 2 * pi) : phi;
    float phi_norm = phi * inv_2pi;

    return { phi_norm, theta_norm };
}

__device__ float GetDeltaStep(int stepSize, float tExit)
{
    float totalDiv = 1.01 * ((1 << (stepSize - 1)) - 1);
    return tExit / totalDiv;
}

__device__ float SetFeatureCoords(
    const glm::vec4 &pointInfo, const RenderParametersPayload::LightInfo &light,
    const VolumeKernelData &volume, glm::vec3 *sampleBuffer, int stepNum)
{
    glm::vec3 currPoint{ pointInfo };
    float tExit = volume.GetBound().IntersectUnchecked(currPoint, light.invDir,
                                                       light.dirIsPos);
    assert(tExit >= 0);

    auto deltaStep = GetDeltaStep(stepNum, tExit) * light.dir;
    for (int i = 0; i < stepNum; i++)
    {
        sampleBuffer[i] = volume.ToSampleCoord(currPoint);
        currPoint += deltaStep;
        deltaStep += deltaStep;
    }
    return tExit;
}

__device__ glm::vec3 GetDirectIllum(
    curandState_t *seed, const VolumeKernelData &volume,
    const glm::vec3 &samplePoint, float tExit, const glm::vec3 &viewDir,
    const glm::vec3 &lightDir, const glm::vec3 &albedo, float g, float scale)
{
    float phase = HenyeyGreenstein(dot(viewDir, lightDir), g);
    float tr = volume.GetEstimatedTrDelta(seed, samplePoint,
                                          samplePoint + tExit * lightDir);

    return scale * tr * phase * albedo;
}

__device__ glm::vec3 GetDirectIllumFast(const VolumePayload &payload,
                                        const glm::vec3 &samplePoint,
                                        const glm::vec3 &viewDir,
                                        const glm::vec3 &lightDir,
                                        const glm::vec3 &albedo, float g,
                                        float scale)
{
    float phase = HenyeyGreenstein(dot(viewDir, lightDir), g);
    auto coord = payload.volume.ToSampleCoord(samplePoint);
    float tr = payload.trMap.map.Sample(coord.x, coord.y, coord.z, 0);
    return scale * exp(-tr) * phase * albedo;
}

template<bool FeatureExtractionOnly = false>
__device__ float TinyRPNNInference(const glm::vec4 &samplePointInfo,
                                   const glm::vec3 &viewDir,
                                   const VolumePayload &volumePayload,
                                   const NetworkGlobalPayload &networkPayload,
                                   const RenderParametersPayload &paramPayload,
                                   NetworkFloat *result)
{
    // Unfortunately, nvcc seems to wrongly optimize code so that when we change
    // it to 'int stepNum = networkPayload.networkNum - 1', the program behavior
    // is completely wrong (only in release mode). And we can NEVER find even
    // a single part that writes out of bound, with many unit tests verifying
    // the related code. Also, we use EVERY tool in compute sanitizer, without
    // finding ANY error too. Therefore, we HAVE TO regard it as nvcc bug, at
    // least currently.
    int stepNum = 4;
    assert(stepNum == networkPayload.networkNum - 1);

    constexpr int featurePart = 5;
    int dims[featurePart] = { networkPayload.baseEncoding.GetFeatureDim(),
                              networkPayload.shEncoding.GetFeatureDim(),
                              networkPayload.viewEncoding.GetFeatureDim(),
                              networkPayload.hgEncoding.GetFeatureDim(),
                              networkPayload.albedoEncoding.GetFeatureDim() };
    int offsets[featurePart] = { dims[0] };
    for (int i = 1; i < featurePart; i++)
        offsets[i] = offsets[i - 1] + dims[i];
    auto lightDim = dims[3];

    // stepNum == trMapSampleDim
    auto totalFeatureDim = offsets[featurePart - 1] + stepNum;
    assert(totalFeatureDim <= FEATURE_DIM_UPPER_BOUND);
    assert(stepNum <= STEP_UPPER_BOUND);
    assert(lightDim <= LIGHT_DIM_UPPER_BOUND);

    glm::vec3 featureCoords[STEP_UPPER_BOUND],
        swizzledFeatureCoords[STEP_UPPER_BOUND];
    float tExit =
        SetFeatureCoords(samplePointInfo, paramPayload.light,
                         volumePayload.volume, featureCoords, stepNum);
    for (int i = 0; i < stepNum; i++)
    {
        swizzledFeatureCoords[i] =
            glm::vec3{ featureCoords[i].z, featureCoords[i].y,
                       featureCoords[i].x };
    }

    NetworkFloat feature[FEATURE_DIM_UPPER_BOUND];

    [[maybe_unused]] NetworkFloat inputFeatureBuffer[FEATURE_DIM_UPPER_BOUND],
        outputFeatureBuffer[3][FEATURE_DIM_UPPER_BOUND]{ 0 },
        tempBuffer[LIGHT_DIM_UPPER_BOUND];

    auto lightDir = paramPayload.light.dir;
    auto viewCoord = ToNormalizedSphericalCoord(viewDir),
         lightCoord = ToNormalizedSphericalCoord(lightDir);
    CHECK_COORD(viewCoord);
    CHECK_COORD(lightCoord);

    // float cosCoord = (glm::dot(viewDir, lightDir) + 1) / 2;
    float cosCoord = (glm::dot(viewDir, lightDir) * 0.98f + 1) / 2;
    // float cosCoord = acos(glm::dot(viewDir, lightDir)) * inv_pi * 0.98f;
    auto hgCoord = glm::vec2{ paramPayload.g, cosCoord };
    CHECK_COORD(hgCoord);

    glm::vec3 albedo = paramPayload.albedo;
    glm::vec3 albedoCoord = (glm::exp(albedo) - 1.0f) / (e - 0.99f);
    CHECK_COORD(albedoCoord);

    float scale = volumePayload.volume.GetInvMaxDensity();
    auto lightPtr = feature + offsets[2];
    for (int i = 0; i < stepNum; i++)
    {
        networkPayload.baseEncoding.Sample<3>(feature, dims[0], i,
                                              swizzledFeatureCoords[i]);
        // For perf SH base grid:
        // networkPayload.shEncoding.Sample<3>(feature, dims[0], i,
        //                                     swizzledFeatureCoords[i],
        //                                     lightDir, 1.0f);

        // Low-freq: always sample at current position.
        networkPayload.shEncoding.Sample<3>(feature + offsets[0], dims[1], i,
                                            swizzledFeatureCoords[0], lightDir,
                                            paramPayload.g);
        if (networkPayload.viewEncDim == 2)
        {
            networkPayload.viewEncoding.Sample<2>(feature + offsets[1], dims[2],
                                                  i, viewCoord);
        }
        else
        {
            networkPayload.viewEncoding.Sample<3>(feature + offsets[1], dims[2],
                                                  i, (viewDir + 1.0f) / 2.0f);
        }

        networkPayload.hgEncoding.Sample<2>(lightPtr, lightDim, i, hgCoord);
        if (!paramPayload.excludeLightEncoding)
        {
            networkPayload.lightEncoding.Sample<2>(tempBuffer, lightDim, i,
                                                   lightCoord);
            for (int j = 0; j < lightDim; j++)
                lightPtr[j] *= tempBuffer[j];
        }
        for (int j = 0; j < stepNum; j++)
        {
            auto currCoord = featureCoords[j];
            float tr = volumePayload.trMap.map.Sample(currCoord.x, currCoord.y,
                                                      currCoord.z, i);
            *(feature + offsets[4] + j) = tr * scale;
        }

        if constexpr (FeatureExtractionOnly)
        {
            for (int i = 0; i < offsets[3]; i++)
                result[i] = feature[i];

            auto currBuffer = result + offsets[3];
            for (int channel = 0; channel < 3; channel++)
            {
                networkPayload.albedoEncoding.Sample<2>(
                    currBuffer, dims[4], i,
                    glm::vec2{ albedoCoord[channel], cosCoord });
                currBuffer += dims[4];
            }

            for (int j = 0; j < stepNum; j++)
            {
                auto currCoord = featureCoords[j];
                float tr = volumePayload.trMap.map.Sample(
                    currCoord.x, currCoord.y, currCoord.z, i);
                currBuffer[j] = tr * scale;
            }

            result = currBuffer + stepNum;
        }
        else
        {
            for (int channel = 0; channel < 3; channel++)
            {
                networkPayload.albedoEncoding.Sample<2>(
                    feature + offsets[3], dims[4], i,
                    glm::vec2{ albedoCoord[channel], cosCoord });

                for (int j = 0; j < totalFeatureDim; j++)
                {
                    inputFeatureBuffer[j] =
                        outputFeatureBuffer[channel][j] + feature[j];
                    outputFeatureBuffer[channel][j] = NetworkFloat{ 0 };
                }

                networkPayload.networks[i].InferenceWithConcat(
                    inputFeatureBuffer, outputFeatureBuffer[channel]);
            }
        }
    }

    if constexpr (!FeatureExtractionOnly)
    {
        auto &finalNetwork = networkPayload.networks[stepNum];
        for (int channel = 0; channel < 3; channel++)
        {
            finalNetwork.Inference(outputFeatureBuffer[channel],
                                   result + channel);
        }
    }
    return tExit;
}

__device__ glm::vec3 Denoise(
    glm::vec3 position, const RenderParametersPayload::TemporalInfo &lastInfo,
    glm::vec3 radianceBeforeDenoise)
{
    if (lastInfo.frameBuffer.bufferPtr == nullptr)
        return radianceBeforeDenoise;

    auto reprojectionDir = position - lastInfo.camera.position;
    float coeff = 1 / glm::dot(reprojectionDir, lastInfo.camera.forward);
    auto u = (glm::dot(reprojectionDir, lastInfo.camera.up) * coeff + 1) / 2;
    auto v = (1 - glm::dot(reprojectionDir, lastInfo.camera.right) * coeff) / 2;
    if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1))
    {
        return radianceBeforeDenoise;
    }

    auto lastResX = lastInfo.frameBuffer.size.x,
         lastResY = lastInfo.frameBuffer.size.y;
    int x = min(int(u * lastResX), lastResX - 1),
        y = min(int(v * lastResY), lastResY - 1);

    auto lastIdx = x * lastResY + y;
    auto lastPtr = lastInfo.frameBuffer.bufferPtr + lastIdx * 3;
    return glm::mix(glm::vec3{ lastPtr[0], lastPtr[1], lastPtr[2] },
                    radianceBeforeDenoise, lastInfo.lerpCoeff);
}

} // namespace

static __global__ void RenderKernel(VolumePayload volumePayload,
                                    NetworkGlobalPayload networkPayload,
                                    RenderParametersPayload paramPayload)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x,
         y = blockIdx.y * blockDim.y + threadIdx.y;
    auto size = paramPayload.frameBuffer.size;
    if (x >= size.x || y >= size.y)
        return;

    auto idx = x * size.y + y;
    auto dstChannelNum = paramPayload.frameBuffer.channelNum;
    auto dst = paramPayload.frameBuffer.bufferPtr + dstChannelNum * idx;
    curandState state;
    InitRand(&state, paramPayload.randomOffset);

    float u = (x + curand_uniform(&state)) / size.x,
          v = (y + curand_uniform(&state)) / size.y;

    const auto &camera = paramPayload.camera;
    glm::vec3 dir = camera.forward + (camera.up * (u * 2 - 1)) -
                    (camera.right * (v * 2 - 1));
    dir = normalize(dir);

    glm::vec4 samplePointInfo =
        volumePayload.volume.GetSamplePoint(&state, camera.position, dir);

    if (samplePointInfo.w < 0)
    {
        glm::vec4 color{ paramPayload.skybox.SampleTexture(dir), 1.0f };
        for (int i = 0; i < dstChannelNum; i++)
            dst[i] = color[i];
        return;
    }

    NetworkFloat resultBuffer[3]{};
    TinyRPNNInference(samplePointInfo, dir, volumePayload, networkPayload,
                      paramPayload, resultBuffer);
    glm::vec3 pred{ resultBuffer[0], resultBuffer[1], resultBuffer[2] };
    pred = glm::max(glm::exp(pred) - 1.0f, glm::vec3{ 0 });

    glm::vec3 denoisedPred =
        Denoise(samplePointInfo, paramPayload.lastFrameInfo, pred);
    glm::vec4 result{ ACES(denoisedPred), 1.0f };

    for (int i = 0; i < dstChannelNum; i++)
        dst[i] = result[i];
    return;
}

static __global__ void DebugKernel(VolumePayload volumePayload,
                                   NetworkGlobalPayload networkPayload,
                                   RenderParametersPayload paramPayload)
{
    glm::vec3 dir{ 0.4, 0.6, 0.69282032303f };
    NetworkFloat resultBuffer[240]{};
    TinyRPNNInference<true>(glm::vec4{ -0.4, -0.3, -0.2, 1.0 }, dir,
                            volumePayload, networkPayload, paramPayload,
                            resultBuffer);

    for (int i = 0; i < 240; i++)
        printf("%f ", (float)resultBuffer[i]);

    return;
}

extern void DispatchRenderKernel(const VolumePayload &volumePayload,
                                 const NetworkGlobalPayload &networkPayload,
                                 const RenderParametersPayload &paramPayload)
{
    dim3 dimBlock(8, 4);
    dim3 dimGrid(RoundUpDiv(paramPayload.frameBuffer.size.x, dimBlock.x),
                 RoundUpDiv(paramPayload.frameBuffer.size.y, dimBlock.y));

    RenderKernel<<<dimGrid, dimBlock>>>(volumePayload, networkPayload,
                                        paramPayload);
    CheckCUDAError(cudaGetLastError());
}

static __global__ void RenderFeatureKernel(VolumePayload volumePayload,
                                           NetworkGlobalPayload networkPayload,
                                           RenderParametersPayload paramPayload,
                                           GBufferPayload gBufferPayload,
                                           NetworkFloat *dstBuffer,
                                           int featureDim, half *radianceBuffer)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x,
         y = blockIdx.y * blockDim.y + threadIdx.y;
    auto size = paramPayload.frameBuffer.size;
    if (x >= size.x || y >= size.y)
        return;

    auto idx = x * size.y + y;
    auto dst = dstBuffer + featureDim * (networkPayload.networkNum - 1) * idx;
    curandState state;
    InitRand(&state, paramPayload.randomOffset);

    float u = (x + curand_uniform(&state)) / size.x,
          v = (y + curand_uniform(&state)) / size.y;

    const auto &camera = paramPayload.camera;
    glm::vec3 dir = camera.forward + (camera.up * (u * 2 - 1)) -
                    (camera.right * (v * 2 - 1));
    dir = normalize(dir);

    glm::vec4 samplePointInfo =
        volumePayload.volume.GetSamplePoint(&state, camera.position, dir);

    gBufferPayload.position[idx] = samplePointInfo;
    gBufferPayload.viewDir[idx] = dir;
    if (samplePointInfo.w < 0)
        return;

    paramPayload.albedo *= volumePayload.volume.GetAlbedo(samplePointInfo);
    float tExit = TinyRPNNInference<true>(samplePointInfo, dir, volumePayload,
                                          networkPayload, paramPayload, dst);
    // We accumulate direct illum inside tcnn kernel directly.
    glm::vec3 directIllum;
    if (paramPayload.fastDirectIllum)
    {
        directIllum = GetDirectIllumFast(
            volumePayload, samplePointInfo, dir, paramPayload.light.dir,
            paramPayload.albedo, paramPayload.g, paramPayload.illumScale);
    }
    else
    {
        directIllum =
            GetDirectIllum(&state, volumePayload.volume, samplePointInfo, tExit,
                           dir, paramPayload.light.dir, paramPayload.albedo,
                           paramPayload.g, paramPayload.illumScale);
    }
    directIllum *= paramPayload.light.color;
    for (int i = 0; i < 3; i++)
        radianceBuffer[idx * 3 + i] = directIllum[i];
    return;
}

static __global__ void DenoiseKernel(
    RenderParametersPayload::FrameBufferInfo dst, EnvironmentMap::Proxy skybox,
    GBufferPayload gBuffer, RenderParametersPayload::TemporalInfo lastFrameInfo,
    RenderParametersPayload::ToneMapping toneMapping,
    const half *radianceBuffer)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x,
         y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst.size.x || y >= dst.size.y)
        return;

    auto idx = x * dst.size.y + y;
    auto dstPtr = dst.bufferPtr + idx * dst.channelNum;
    auto swapPtr = lastFrameInfo.swapBuffer + idx * 3;

    auto position = gBuffer.position[idx];
    glm::vec3 pred;
    if (position.w < 0)
    {
        auto dir = gBuffer.viewDir[idx];
        pred = skybox.SampleTexture(dir);
        // for (int i = 0; i < dst.channelNum; i++)
        //     dstPtr[i] = 0.0f;
        // return;
    }
    else
    {
        // for (int i = 0; i < dst.channelNum; i++)
        //     dstPtr[i] = 1.0f;
        // return;
        auto radiancePtr = radianceBuffer + idx * 3;
        pred = glm::vec3{ radiancePtr[0], radiancePtr[1], radiancePtr[2] };
    }

    glm::vec3 denoisedPred = Denoise(position, lastFrameInfo, pred);
    for (int i = 0; i < 3; i++)
        swapPtr[i] = denoisedPred[i];

    if (toneMapping == RenderParametersPayload::ToneMapping::Gamma)
        denoisedPred = Gamma(denoisedPred);
    else if (toneMapping == RenderParametersPayload::ToneMapping::ACES)
        denoisedPred = ACES(denoisedPred);

    glm::vec4 result{ denoisedPred, 1.0f };
    for (int i = 0; i < dst.channelNum; i++)
        dstPtr[i] = result[i];
    return;
}

// Have only very little acceleration.
static std::uint32_t CombineAlbedoHint(glm::vec3 albedo)
{
    const bool g_eq_r = (albedo[1] == albedo[0]);
    const bool b_eq_r = (albedo[2] == albedo[0]);
    const bool b_eq_g = (albedo[2] == albedo[1]);

    const std::uint32_t greenCode = !g_eq_r;
    const std::uint32_t blueCode = b_eq_r ? 0u : (!g_eq_r && b_eq_g) ? 1u : 2u;

    return (blueCode << 8) | greenCode;
}

static constexpr std::uint32_t GetDefaultAlbedoHint()
{
    return (2 << 8) | 1;
}

extern void DispatchTCNNRenderKernel(
    const VolumePayload &volumePayload,
    const NetworkGlobalPayload &networkPayload,
    const RenderParametersPayload &paramPayload, const TCNNPayload &tcnnPayload,
    const GBufferPayload &gBufferPayload, EnvironmentBaking &envBaking)
{
    dim3 dimBlock(8, 4);
    dim3 dimGrid(RoundUpDiv(paramPayload.frameBuffer.size.x, dimBlock.x),
                 RoundUpDiv(paramPayload.frameBuffer.size.y, dimBlock.y));

    RenderFeatureKernel<<<dimGrid, dimBlock>>>(
        volumePayload, networkPayload, paramPayload, gBufferPayload,
        tcnnPayload.featureBuffer, tcnnPayload.featureDim,
        tcnnPayload.radianceBuffer);
    CheckCUDAError(cudaGetLastError());
    bool hasAlbedoGrid = volumePayload.volume.HasAlbedoGrid();
    auto lightColor = paramPayload.light.color;
    ExternalTCNNInference(
        tcnnPayload.handle, tcnnPayload.featureBuffer,
        tcnnPayload.radianceBuffer, gBufferPayload.position, tcnnPayload.size,
        hasAlbedoGrid ? GetDefaultAlbedoHint()
                      : CombineAlbedoHint(paramPayload.albedo),
        Color3{ lightColor.r, lightColor.g, lightColor.b });

    if (envBaking.Valid() && paramPayload.enableSkyboxBaking)
    {
        if (hasAlbedoGrid)
        {
            std::cerr << "WARNING: Environment Baking hasn't support albedo "
                         "grid yet in current impl. It's omitted now.\n";
        }

        auto unpaddedSize =
            static_cast<std::size_t>(paramPayload.frameBuffer.size.x) *
            paramPayload.frameBuffer.size.y;
        envBaking.Inference(EnvBakingPayload::ExternalInfo{
            .radianceBuffer = tcnnPayload.radianceBuffer,
            .positionBuffer = gBufferPayload.position,
            .viewDirBuffer = gBufferPayload.viewDir,
            .unpaddedSize = unpaddedSize,
            .exposure = paramPayload.skybox.Exposure(),
            .g = paramPayload.g,
            .albedo = paramPayload.albedo,
            .volumeBound = volumePayload.volume.GetBound(),
        });
    }

    // Then do denoise.
    DenoiseKernel<<<dimGrid, dimBlock>>>(
        paramPayload.frameBuffer, paramPayload.skybox, gBufferPayload,
        paramPayload.lastFrameInfo, paramPayload.toneMapping,
        tcnnPayload.radianceBuffer);
    CheckCUDAError(cudaGetLastError());
}