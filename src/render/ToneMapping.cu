/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "ToneMapping.cuh"

namespace
{
__device__ glm::vec3 unity_to_ACES(glm::vec3 x)
{
    glm::mat3 sRGB_2_AP0 = { { 0.4397010, 0.3829780, 0.1773350 },
                             { 0.0897923, 0.8134230, 0.0967616 },
                             { 0.0175440, 0.1115440, 0.8707040 } };
    x = glm::transpose(sRGB_2_AP0) * x;
    return x;
}

__device__ glm::vec3 ACES_to_ACEScg(glm::vec3 x)
{
    glm::mat3 AP0_2_AP1_MAT = { { 1.4514393161, -0.2365107469, -0.2149285693 },
                                { -0.0765537734, 1.1762296998, -0.0996759264 },
                                { 0.0083161484, -0.0060324498, 0.9977163014 } };
    return glm::transpose(AP0_2_AP1_MAT) * x;
}

__device__ glm::vec3 XYZ_2_xyY(glm::vec3 XYZ)
{
    float divisor = max(dot(XYZ, { 1, 1, 1 }), 1e-4);
    return glm::vec3{ XYZ.x / divisor, XYZ.y / divisor, XYZ.y };
}

__device__ glm::vec3 xyY_2_XYZ(glm::vec3 xyY)
{
    float m = xyY.z / max(xyY.y, 1e-4f);
    glm::vec3 XYZ = glm::vec3{ xyY.x, xyY.z, (1.0f - xyY.x - xyY.y) };
    XYZ.x *= m;
    XYZ.z *= m;
    return XYZ;
}

__device__ glm::vec3 darkSurround_to_dimSurround(glm::vec3 linearCV)
{
    glm::mat3 AP1_2_XYZ_MAT =
        glm::mat3{ { 0.6624541811, 0.1340042065, 0.1561876870 },
                   { 0.2722287168, 0.6740817658, 0.0536895174 },
                   { -0.0055746495, 0.0040607335, 1.0103391003 } };
    glm::vec3 XYZ = glm::transpose(AP1_2_XYZ_MAT) * linearCV;

    glm::vec3 xyY = XYZ_2_xyY(XYZ);
    xyY.z = min(max(xyY.z, 0.0), 65504.0);
    xyY.z = pow(xyY.z, 0.9811f);
    XYZ = xyY_2_XYZ(xyY);

    glm::mat3 XYZ_2_AP1_MAT = { { 1.6410233797, -0.3248032942, -0.2364246952 },
                                { -0.6636628587, 1.6153315917, 0.0167563477 },
                                { 0.0117218943, -0.0082844420, 0.9883948585 } };
    return glm::transpose(XYZ_2_AP1_MAT) * XYZ;
}
} // namespace

__device__ glm::vec3 ACES(glm::vec3 color)
{
    glm::mat3 AP1_2_XYZ_MAT =
        glm::mat3{ { 0.6624541811, 0.1340042065, 0.1561876870 },
                   { 0.2722287168, 0.6740817658, 0.0536895174 },
                   { -0.0055746495, 0.0040607335, 1.0103391003 } };

    glm::vec3 aces = unity_to_ACES(color);

    glm::vec3 AP1_RGB2Y = glm::vec3{ 0.272229, 0.674082, 0.0536895 };

    glm::vec3 acescg = ACES_to_ACEScg(aces);
    float tmp = dot(acescg, AP1_RGB2Y);
    acescg = glm::mix(glm::vec3{ tmp, tmp, tmp }, acescg, 0.96f);
    const float a = 278.5085;
    const float b = 10.7772;
    const float c = 293.6045;
    const float d = 88.7122;
    const float e = 80.6889;
    glm::vec3 x = acescg;
    glm::vec3 rgbPost = (x * (x * a + b)) / (x * (x * c + d) + e);
    glm::vec3 linearCV = darkSurround_to_dimSurround(rgbPost);
    tmp = dot(linearCV, AP1_RGB2Y);
    linearCV = glm::mix(glm::vec3{ tmp, tmp, tmp }, linearCV, 0.93f);
    glm::vec3 XYZ = glm::transpose(AP1_2_XYZ_MAT) * linearCV;
    glm::mat3 D60_2_D65_CAT = { { 0.98722400, -0.00611327, 0.0159533 },
                                { -0.00759836, 1.00186000, 0.0053302 },
                                { 0.00307257, -0.00509595, 1.0816800 } };
    XYZ = glm::transpose(D60_2_D65_CAT) * XYZ;
    glm::mat3 XYZ_2_REC709_MAT = {
        { 3.2409699419, -1.5373831776, -0.4986107603 },
        { -0.9692436363, 1.8759675015, 0.0415550574 },
        { 0.0556300797, -0.2039769589, 1.0569715142 }
    };
    linearCV = glm::transpose(XYZ_2_REC709_MAT) * XYZ;

    return Gamma(linearCV);
}