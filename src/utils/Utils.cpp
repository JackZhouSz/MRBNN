/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "Utils.hpp"

#include <fstream>
#include <iostream>

void LogError(const char *errMsg)
{
    std::cerr << "Error: " << errMsg << "\n";
}

void LogError(std::string_view errMsg)
{
    std::cerr << "Error: " << errMsg << "\n";
}

void CheckCUDAError(cudaError_t err)
{
    if (err != cudaError::cudaSuccess)
    {
        LogError(cudaGetErrorString(err));
        throw std::runtime_error{ cudaGetErrorName(err) };
    }
}

std::ostringstream ReadAllFromFile(const std::filesystem::path &path,
                                   bool addTrailingZero, bool isBinary)
{
    std::ifstream fin{ path, isBinary ? std::ios::binary : std::ios::in };
    fin.exceptions(std::ios::failbit | std::ios::badbit);

    std::ostringstream buffer;
    fin >> buffer.rdbuf();
    if (addTrailingZero)
        buffer << '\0';
    return buffer;
}