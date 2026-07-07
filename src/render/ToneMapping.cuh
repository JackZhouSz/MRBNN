/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include <glm/glm.hpp>

inline __device__ glm::vec3 Gamma(glm::vec3 color)
{
    return glm::pow(color, glm::vec3(1.0f / 2.2f));
}

__device__ glm::vec3 ACES(glm::vec3 color);