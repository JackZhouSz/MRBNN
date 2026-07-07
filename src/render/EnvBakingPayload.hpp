/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include "RenderPayload.hpp"
#include "cuda_runtime.h"

struct EnvBakingPayload
{
    struct ExternalInfo
    {
        NetworkFloat *radianceBuffer;
        glm::vec4 *positionBuffer;
        glm::vec3 *viewDirBuffer;
        std::size_t unpaddedSize;

        float exposure;
        float g;
        glm::vec3 albedo;
        // We don't pass the whole volume to minimize dependency; essentially
        // only use ToSampleCoord and IntersectUnchecked.
        Bounds3f volumeBound;
    } info;

    struct Encodings
    {
        EncodingGPUProxy<NetworkFloat> base;
        SHEncodingGPUProxy<NetworkFloat, 2> sh;
        EncodingGPUProxy<NetworkFloat> attrs[3];
        int stepNum;
    } encodings;
};