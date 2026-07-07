/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "EnvironmentBaking.hpp"
#include "Network.hpp"
#include "RenderPayload.hpp"
#include "Utils.hpp"

#define DLL_MACRO_NEED_IMPORT
#include "ExternalTCNN.hpp"

#include <array>
#include <fstream>

struct EnvironmentBaking::Impl
{
    float scale = 1.0f;

    Encoding<NetworkFloat> baseEncoding;
    SphericalEncoding<NetworkFloat, 2> shEncoding;
    std::array<Encoding<NetworkFloat>, 3> attrEncodings;
    int featureDim = 0;
    int stepNum = 0;

    UniqueResource<NetworkFloat *, CUDADeleter> featureBuffer;
    std::size_t size = 0;

    UniqueResource<void *, Deleter<DeleteTCNNHandler>> tcnnHandle;
    void TryResize(std::size_t newSize)
    {
        newSize = RoundUpDiv(newSize, std::size_t{ WARP_SIZE }) * WARP_SIZE;
        if (newSize == size)
            return;

        half *buffer;
        CheckCUDAError(cudaMalloc(&buffer, newSize * featureDim * stepNum *
                                               sizeof(NetworkFloat)));
        featureBuffer.Reset(buffer);
        size = newSize;
    }
};

EnvironmentBaking::EnvironmentBaking(const std::filesystem::path &workDir)
{
    std::ifstream fin{ workDir / "config.json" };
    auto config = nlohmann::json::parse(fin);
    auto GetAttrEncoding = [&](int idx) {
        return Encoding<NetworkFloat>{ workDir / std::format("attr{}.bin", idx),
                                       config["attr_encoding"], 2 };
    };

    impl_.reset(new Impl{
        .baseEncoding = Encoding<NetworkFloat>{ workDir / "base.bin",
                                                config["volume_encoding"], 3 },
        .shEncoding =
            SphericalEncoding<NetworkFloat, 2>{ workDir / "ms",
                                                config["ms_encoding"], 3 },
        .attrEncodings = { GetAttrEncoding(0), GetAttrEncoding(1),
                           GetAttrEncoding(2) },
    });

    auto GetDim = [&config](const auto &key) {
        return config[key]["n_features_per_level"].template get<int>();
    };
    int offsetDim = GetDim("volume_encoding") + GetDim("ms_encoding");
    int albedoDim = GetDim("attr_encoding");
    int featureDim = offsetDim + 3 * albedoDim;
    impl_->featureDim = featureDim;
    impl_->scale = config.value("scale", 1.0f);

    while (std::filesystem::exists(
        workDir / std::format("tcnn.mlp{}.bin", impl_->stepNum)))
    {
        impl_->stepNum++;
    }

    impl_->tcnnHandle.Reloc(NewTCNNHandler(workDir.string().c_str(), featureDim,
                                           offsetDim, albedoDim));
}

EnvironmentBaking::EnvironmentBaking() noexcept = default;
EnvironmentBaking::EnvironmentBaking(EnvironmentBaking &&) noexcept = default;
EnvironmentBaking &EnvironmentBaking::operator=(EnvironmentBaking &&) noexcept =
    default;
EnvironmentBaking::~EnvironmentBaking() = default;

// In corresponding .cu file.
extern void FillEnvFeatureBuffer(const EnvBakingPayload &payload,
                                 NetworkFloat *featureBuffer, int featureDim);

void EnvironmentBaking::Inference(const EnvBakingPayload::ExternalInfo &info)
{
    impl_->TryResize(info.unpaddedSize);
    EnvBakingPayload payload{
        .info = info,
        .encodings = { .base = impl_->baseEncoding.ToProxy(),
                       .sh = impl_->shEncoding.ToProxy(),
                       .attrs = { impl_->attrEncodings[0].ToProxy(),
                                  impl_->attrEncodings[1].ToProxy(),
                                  impl_->attrEncodings[2].ToProxy() },
                       .stepNum = impl_->stepNum },
    };
    float exposure = payload.info.exposure * impl_->scale;
    payload.info.exposure = exposure;
    FillEnvFeatureBuffer(payload, impl_->featureBuffer.Get(),
                         impl_->featureDim);
    // Use impl_->size for it is padded.
    ExternalTCNNInference(impl_->tcnnHandle.Get(), impl_->featureBuffer.Get(),
                          info.radianceBuffer, info.positionBuffer, impl_->size,
                          1 + (2 << 8), Color3{ exposure, exposure, exposure });
    return;
}
