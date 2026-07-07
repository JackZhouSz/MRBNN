/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include <filesystem>
#include <optional>

struct COMContextGuard
{
    COMContextGuard();
    ~COMContextGuard();
};

std::optional<std::filesystem::path> OpenFileDialog();
std::optional<std::filesystem::path> OpenFolderDialog();