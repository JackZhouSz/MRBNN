/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"

int main(int argc, const char *argv[])
{
    if (argc != 3)
    {
        std::cerr
            << "Usage: ViewAdjustment PARAM_CONFIG_PATH DST_CONFIG_ROOT_DIR";
        return 0;
    }

    nlohmann::json paramConfig;
    {
        std::ifstream fin{ argv[1] };
        paramConfig = nlohmann::json::parse(fin);
    }

    auto &cameraPosition =
        paramConfig["cameraPos"].get_ref<nlohmann::json::array_t &>();
    float x = cameraPosition[0], y = cameraPosition[1], z = cameraPosition[2];
    float initRadius = std::hypot(x, y, z);

    std::filesystem::path rootPath{ argv[2] };
    nlohmann::json toolConfig;
    {
        std::ifstream fin{ rootPath / "config.json" };
        toolConfig = nlohmann::json::parse(fin);
    }

    for (int i = 6; i > 0; i--)
    {
        auto newViewRootPath = rootPath / std::format("{}", i);
        // Don't detect return value since we tolerate path to already exist.
        std::filesystem::create_directory(newViewRootPath);

        float scale = (0.1f * i) / initRadius;

        cameraPosition[0] = x * scale, cameraPosition[1] = y * scale,
        cameraPosition[2] = z * scale;

        auto newParamConfigPath = newViewRootPath / "init.json";
        std::ofstream fout{ newParamConfigPath };
        fout << std::setw(4) << paramConfig;
        fout.close();

        toolConfig["config_file"] = newParamConfigPath.string();
        toolConfig["out_dir"] = newViewRootPath.string();
        fout.open(newViewRootPath / "config.json");
        fout << std::setw(4) << toolConfig;
    }

    return 0;
}