/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include "EnvironmentMap.hpp"
#include "Texture2D.hpp"
#include "Volume.hpp"

struct RenderParameters
{
    struct FrameBufferInfo
    {
        glm::u32vec2 size;
        float *bufferPtr;
        unsigned int channelNum;
    } frameBuffer;

    int frameNum = 0;

    float g = 0.857f;
    glm::vec3 albedo = glm::vec3{ 1.0f / 1.001f };

    EnvironmentMap skybox;
    bool enableSkybox = false;

    enum class DensityLinearity : std::uint8_t
    {
        Point,
        Linear,
        Point_MIPMapLinear,
        Linear_MIPMapLinear
    } densityInterpolation = DensityLinearity::Point;
    bool recreateDensityTexture = false;

    bool refreshTrMap = false;
    int trMapSampleLevel = 0;
    int trMapMarchingSteps = 0;

    // Initial position for cloud0
    glm::vec3 cameraPos{ 0.67085f, -0.03808f, -0.04856f };
    glm::vec3 lightDir{ 0.34281, 0.70711, 0.61845 };
    glm::vec3 lightColor{ 1.0f };

    enum class ToneMapping
    {
        None,
        Gamma,
        ACES
    } toneMapping = ToneMapping::ACES;

    enum class Denoise
    {
        None,
        Unbiased,
        VisualPlausible
    } denoise = Denoise::VisualPlausible;

    enum class Compatibility
    {
        Normal,
        MRPNN,
        NoDirectIllum
    } compatibility = Compatibility::Normal;
    bool excludeLightEncoding = true;
    bool fastDirectIllum = false;
    bool enableSkyboxBaking = false;
};

class RenderInterface
{
public:
    struct NetworkData;
    struct AuxData;

    RenderInterface(const std::filesystem::path &workDir);
    RenderInterface(const RenderInterface &) = delete;
    RenderInterface &operator=(const RenderInterface &) = delete;
    RenderInterface(RenderInterface &&) noexcept; // Pimpl.
    RenderInterface &operator=(RenderInterface &&) noexcept;
    ~RenderInterface();

    void Render(const RenderParameters &params);

protected:
    template<typename T, typename Func>
    void RenderBase_(const RenderParameters &params, Func renderFunc);

    GPUSimpleVolume volume_;
    TransmittanceMap transmittanceMap_;
    std::unique_ptr<NetworkData> networkData_;
    std::unique_ptr<AuxData> auxData_;
};

class RenderInterfaceWithTCNN : public RenderInterface
{
public:
    struct TCNNData;

    RenderInterfaceWithTCNN(const std::filesystem::path &workDir);
    RenderInterfaceWithTCNN(const RenderInterfaceWithTCNN &) = delete;
    RenderInterfaceWithTCNN &operator=(const RenderInterfaceWithTCNN &) =
        delete;
    RenderInterfaceWithTCNN(RenderInterfaceWithTCNN &&) noexcept; // Pimpl.
    RenderInterfaceWithTCNN &operator=(RenderInterfaceWithTCNN &&) noexcept;
    ~RenderInterfaceWithTCNN();

    void Render(const RenderParameters &params);
    void EquipEnvBaking(const std::filesystem::path *workDir);

private:
    std::unique_ptr<TCNNData> tcnnData_;
};