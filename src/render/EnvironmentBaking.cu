/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "EnvBakingPayload.hpp"
#include "Utils.hpp"

#include <numbers>

#define STEP_UPPER_BOUND 8

namespace
{

static constexpr float e = std::numbers::e_v<float>;

__device__ float GetDeltaStep(int stepSize, float tExit)
{
    float totalDiv = 1.01 * ((1 << (stepSize - 1)) - 1);
    return tExit / totalDiv;
}

__device__ void SetFeatureCoords(glm::vec3 currPoint, const glm::vec3 &viewDir,
                                 const Bounds3f &bound, glm::vec3 *sampleBuffer,
                                 int stepNum)
{
    float tExit = bound.IntersectUnchecked(
        currPoint, 1.0f / viewDir,
        glm::ivec3{ glm::greaterThanEqual(viewDir, glm::vec3{ 0 }) });
    assert(tExit >= 0);

    auto deltaStep = GetDeltaStep(stepNum, tExit) * viewDir;
    for (int i = 0; i < stepNum; i++)
    {
        auto coord = bound.Offset(currPoint);
        sampleBuffer[i] = coord;
        currPoint += deltaStep;
        deltaStep += deltaStep;
    }
    return;
}

template<typename T>
__device__ T ExpMapping(T vec)
{
    return (glm::exp(vec) - 1.0f) / (e - 0.99f);
}

} // namespace

__global__ void FillEnvFeatureBufferKernel(EnvBakingPayload payload,
                                           NetworkFloat *featureBuffer,
                                           int featureDim)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= payload.info.unpaddedSize)
        return;

    constexpr int featurePart = 3;
    const unsigned int dims[featurePart] = {
        payload.encodings.base.GetFeatureDim(),
        payload.encodings.sh.GetFeatureDim(),
        payload.encodings.attrs[0].GetFeatureDim()
    };

    auto position = payload.info.positionBuffer[idx];
    if (position.w < 0)
        return;

    auto dir = payload.info.viewDirBuffer[idx];
    // int stepNum = payload.encodings.stepNum;
    int stepNum = 4;
    auto dstPtr = featureBuffer + featureDim * stepNum * idx;

    glm::vec3 featureCoords[STEP_UPPER_BOUND];
    SetFeatureCoords(position, dir, payload.info.volumeBound, featureCoords,
                     stepNum);
    auto albedoCoord = ExpMapping(payload.info.albedo);
    float gCoord = ExpMapping(payload.info.g);

    unsigned int offset = dims[0] + dims[1];
    for (int i = 0; i < stepNum; i++)
    {
        payload.encodings.base.Sample<3>(dstPtr, dims[0], i, featureCoords[i]);
        payload.encodings.sh.Sample<3>(dstPtr + dims[0], dims[1], i,
                                       featureCoords[0], dir, payload.info.g);
        for (int j = 0; j < 3; j++)
        {
            payload.encodings.attrs[j].Sample<2>(
                dstPtr + offset + dims[2] * j, dims[2], i,
                glm::vec2{ albedoCoord[j], gCoord });
        }
        dstPtr += featureDim;
    }
    return;
}

__managed__ float position[4]{ 0.45 - 0.5, 0.55 - 0.5, 0.50 - 0.5, 1.0 };
__managed__ float viewDir[3]{ 0.6, 0.8, 0.0 };
__managed__ half radiances[3];
__managed__ half features[40 * 4];

void FillEnvFeatureBuffer(const EnvBakingPayload &payload,
                          NetworkFloat *featureBuffer, int featureDim)
{
    constexpr std::size_t threadNum = 128;
    auto groupNum = RoundUpDiv(payload.info.unpaddedSize, threadNum);
    FillEnvFeatureBufferKernel<<<groupNum, threadNum>>>(payload, featureBuffer,
                                                        featureDim);
    return;
}