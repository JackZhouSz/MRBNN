/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
#include "render/RenderInterface.hpp"
#include <filesystem>

class ConfiguationLoader
{
public:
    static void Load(const std::filesystem::path &path,
                     RenderParameters &params);
    static void Save(const std::filesystem::path &path,
                     const RenderParameters &params);

    static std::string TryLoadSkybox(const std::filesystem::path &path,
                                     RenderParameters &params);
};

namespace Description
{
// Oh my gosh, what a pain without reflection in C++26.
inline const char *s_toneMapping[] = { "None", "Gamma", "ACES" };
inline const char *s_denoise[] = { "None", "Unbiased", "Visual Quality" };
inline const char *s_interpolation[] = { "Point", "Linear",
                                         "Point_MIPMapLinear",
                                         "Linear_MIPMapLinear" };
inline const char *s_lightCompatibility[] = { "Normal", "MRPNN", "Disable" };

} // namespace Description