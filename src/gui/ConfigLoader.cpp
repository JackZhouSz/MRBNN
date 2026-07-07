/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "ConfigLoader.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;
namespace glm
{
void to_json(json &j, glm::vec3 p)
{
    j = json::array_t{ p[0], p[1], p[2] };
}

void from_json(const json &j, glm::vec3 &p)
{
    for (int i = 0; i < 3; i++)
        j[i].get_to(p[i]);
}
} // namespace glm

using namespace Description;

namespace
{
NLOHMANN_JSON_SERIALIZE_ENUM(
    RenderParameters::DensityLinearity,
    { { RenderParameters::DensityLinearity::Point, s_interpolation[0] },
      { RenderParameters::DensityLinearity::Linear, s_interpolation[1] },
      { RenderParameters::DensityLinearity::Point_MIPMapLinear,
        s_interpolation[2] },
      { RenderParameters::DensityLinearity::Linear_MIPMapLinear,
        s_interpolation[3] } })
static_assert(std::size(s_interpolation) == 4);

NLOHMANN_JSON_SERIALIZE_ENUM(
    RenderParameters::ToneMapping,
    { { RenderParameters::ToneMapping::None, s_toneMapping[0] },
      { RenderParameters::ToneMapping::Gamma, s_toneMapping[1] },
      { RenderParameters::ToneMapping::ACES, s_toneMapping[2] } })
static_assert(std::size(s_toneMapping) == 3);

NLOHMANN_JSON_SERIALIZE_ENUM(
    RenderParameters::Denoise,
    { { RenderParameters::Denoise::None, s_denoise[0] },
      { RenderParameters::Denoise::Unbiased, s_denoise[1] },
      { RenderParameters::Denoise::VisualPlausible, s_denoise[2] } })
static_assert(std::size(s_denoise) == 3);

NLOHMANN_JSON_SERIALIZE_ENUM(
    RenderParameters::Compatibility,
    { { RenderParameters::Compatibility::Normal, s_lightCompatibility[0] },
      { RenderParameters::Compatibility::MRPNN, s_lightCompatibility[1] },
      { RenderParameters::Compatibility::NoDirectIllum,
        s_lightCompatibility[2] } })
static_assert(std::size(s_lightCompatibility) == 3);

struct ConfigHint
{
    float g = 0.857f;
    glm::vec3 albedo = glm::vec3{ 1.0f / 1.001f };

    std::string skyboxPath;
    float skyboxSampleLevel = 0.0f;
    bool enableSkybox = false;

    RenderParameters::DensityLinearity densityInterpolation =
        RenderParameters::DensityLinearity::Point;
    int trMapSampleLevel = 0;
    int trMapMarchingSteps = 0;

    glm::vec3 cameraPos{ 0.67085f, -0.03808f, -0.04856f };
    glm::vec3 lightDir{ 0.34281f, 0.70711f, 0.61845f };
    glm::vec3 lightColor{ 1.0f };

    RenderParameters::ToneMapping toneMapping =
        RenderParameters::ToneMapping::ACES;
    RenderParameters::Denoise denoise =
        RenderParameters::Denoise::VisualPlausible;
    RenderParameters::Compatibility compatibility =
        RenderParameters::Compatibility::Normal;
    bool excludeLightEncoding = true;
};

// clang-format off
#define JSON_FROM_ALLOW_ABSENSE(v1) if(!nlohmann_json_j.is_null()) if (auto it = nlohmann_json_j.find(#v1); it != nlohmann_json_j.end()) nlohmann_json_t.v1 = *it;

#define DEFINE_TYPE_NON_INTRUSIVE_ALLOW_ABSENSE(Type, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(JSON_FROM_ALLOW_ABSENSE, __VA_ARGS__)) }
// clang-format on

// TODO: we haven't really add skybox yet.
#define ALL_MEMBERS                                                           \
    g, albedo, densityInterpolation, trMapSampleLevel, trMapMarchingSteps,    \
        cameraPos, lightDir, lightColor, toneMapping, denoise, compatibility, \
        excludeLightEncoding

DEFINE_TYPE_NON_INTRUSIVE_ALLOW_ABSENSE(ConfigHint, ALL_MEMBERS);

#define MEMBER_ASSIGN_FROM(Member) params.Member = hint.Member,
#define MEMBER_WISE_ASSIGN_FROM_IMPL(...)                         \
    do                                                            \
    {                                                             \
        NLOHMANN_JSON_EXPAND(                                     \
            NLOHMANN_JSON_PASTE(MEMBER_ASSIGN_FROM, __VA_ARGS__)) \
        (void)0;                                                  \
    } while (0)

#define MEMBER_ASSIGN_TO(Member) hint.Member = params.Member,
#define MEMBER_WISE_ASSIGN_TO_IMPL(...)                         \
    do                                                          \
    {                                                           \
        NLOHMANN_JSON_EXPAND(                                   \
            NLOHMANN_JSON_PASTE(MEMBER_ASSIGN_TO, __VA_ARGS__)) \
        (void)0;                                                \
    } while (0);

#define MEMBER_WISE_ASSIGN_FROM MEMBER_WISE_ASSIGN_FROM_IMPL(ALL_MEMBERS)
#define MEMBER_WISE_ASSIGN_TO MEMBER_WISE_ASSIGN_TO_IMPL(ALL_MEMBERS)

void LoadConfigImpl(const std::filesystem::path &path, RenderParameters &params)
{
    std::ifstream fin{ path };
    ConfigHint hint;
    json j = json::parse(fin);
    j.get_to(hint);
    MEMBER_WISE_ASSIGN_FROM;
    // Refresh all directly.
    params.recreateDensityTexture = params.refreshTrMap = true;
    params.frameNum = 0;
    return;
}

void SaveConfigImpl(const std::filesystem::path &path,
                    const RenderParameters &params)
{
    std::ofstream fout{ path };
    ConfigHint hint;
    MEMBER_WISE_ASSIGN_TO;

    json j = hint;
    fout << std::setw(4) << j;
    return;
}

} // namespace

void ConfiguationLoader::Load(const std::filesystem::path &path,
                              RenderParameters &params)
{
    LoadConfigImpl(path, params);
}

void ConfiguationLoader::Save(const std::filesystem::path &path,
                              const RenderParameters &params)
{
    SaveConfigImpl(path, params);
}

std::string ConfiguationLoader::TryLoadSkybox(const std::filesystem::path &path,
                                              RenderParameters &params)
{
    std::ifstream fin{ path };
    if (!fin) // Doesn't exist.
        return {};

    auto config = nlohmann::json::parse(fin);
    auto &imagePath = config["image_path"].get_ref<const std::string &>();
    auto exposure = config.value("exposure", 1.f);
    params.skybox = EnvironmentMap{ imagePath, exposure };

    auto bakingDirIt = config.find("baking_dir");
    if (bakingDirIt == config.end())
    {
        params.enableSkybox = true;
        return {};
    }
    params.enableSkyboxBaking = true;
    return std::move(bakingDirIt.value().get_ref<std::string &>());
}
