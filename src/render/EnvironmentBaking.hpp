/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include "EnvBakingPayload.hpp"

#include <filesystem>

class EnvironmentBaking
{
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    EnvironmentBaking() noexcept;
    EnvironmentBaking(const std::filesystem::path &workDir);
    EnvironmentBaking(const EnvironmentBaking &) = delete;
    EnvironmentBaking &operator=(const EnvironmentBaking &) = delete;
    EnvironmentBaking(EnvironmentBaking &&) noexcept;
    EnvironmentBaking &operator=(EnvironmentBaking &&) noexcept;
    ~EnvironmentBaking();

    bool Valid() const noexcept { return impl_ != nullptr; }
    void Inference(const EnvBakingPayload::ExternalInfo &buffers);
};