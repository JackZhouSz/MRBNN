/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once

#include "EnvironmentMap.hpp"
#include "FeatureExtract.cuh"
#include "Texture2D.hpp"
#include "Volume.cuh"

#define BASE_FEATURE_DIM_UPPER_LIMIT 8
#define HIDDEN_FEATURE_DIM_UPPER_LIMIT 64
#define AGGRESIVE_OMIT_HIDDEN_LAYER 1

using NetworkFloat = half;

template<typename T>
class Encoding;

template<typename T>
class EncodingGPUProxy
{
public:
    static inline constexpr int s_maxLevels = 4;

private:
    T *data_ = nullptr;
    std::size_t resolution_[s_maxLevels];
    std::size_t offsets_[s_maxLevels + 1];
    int featurePerLevel_;

#ifdef DEBUG
    struct
    {
        int inputDims;
    } safetyChecker_;
#endif

    friend class Encoding<T>;

public:
    EncodingGPUProxy() = default;

    __host__ __device__ int GetFeatureDim() const noexcept
    {
        return featurePerLevel_;
    }

#ifdef __CUDACC__
    template<int InputDim, typename U = T>
    __device__ void Sample(U *dst, unsigned int len, int level,
                           glm::vec<InputDim, float> coord) const
    {
#ifdef DEBUG
        assert(safetyChecker_.inputDims == InputDim);
        assert(featurePerLevel_ == len);
#endif
        GetFeature(coord, data_ + offsets_[level] * len,
                   glm::vec<InputDim, int>{ resolution_[level] },
                   offsets_[level + 1] - offsets_[level], len, dst);
    }
#endif
};

template<typename T, std::size_t Order>
class SphericalEncoding;

template<typename T, std::size_t Order>
class SHEncodingGPUProxy
{
public:
    static constexpr inline std::size_t s_shNum = (Order + 1) * (Order + 1);

private:
    T *data_[s_shNum]{};
    std::size_t resolution_[EncodingGPUProxy<T>::s_maxLevels];
    std::size_t offsets_[EncodingGPUProxy<T>::s_maxLevels + 1];
    int featurePerLevel_;

#ifdef DEBUG
    struct
    {
        int inputDims;
    } safetyChecker_;
#endif

    friend class SphericalEncoding<T, Order>;

public:
    SHEncodingGPUProxy() = default;
    __host__ __device__ int GetFeatureDim() const noexcept
    {
        return featurePerLevel_;
    }

#ifdef __CUDACC__
    template<int InputDim, typename U = T>
    __device__ void Sample(U *dst, unsigned int len, int level,
                           glm::vec<InputDim, float> coord, glm::vec3 dir,
                           float aux) const
    {
        float coeffs[s_shNum];
        coeffs[0] = 0.282094791773878;

        if constexpr (Order >= 1)
        {
            coeffs[1] = -0.48860251190292 * dir.y * aux;
            coeffs[2] = 0.48860251190292 * dir.z * aux;
            coeffs[3] = -0.48860251190292 * dir.x * aux;
        }

        if constexpr (Order >= 2)
        {
            auto aux2 = aux * aux;
            coeffs[4] = 1.09254843059208 * dir.x * dir.y * aux2;
            coeffs[5] = -1.09254843059208 * dir.y * dir.z * aux2;
            coeffs[6] =
                (0.94617469575756 * dir.z * dir.z - 0.31539156525252) * aux2;
            coeffs[7] = -1.09254843059208 * dir.x * dir.z * aux2;
            coeffs[8] =
                0.54627421529604 * (dir.x * dir.x - dir.y * dir.y) * aux2;
        }

#ifdef DEBUG
        assert(safetyChecker_.inputDims == InputDim);
        assert(featurePerLevel_ == len);
#endif

        float tempFeature[BASE_FEATURE_DIM_UPPER_LIMIT],
            tempFeature2[BASE_FEATURE_DIM_UPPER_LIMIT]{};
        for (int i = 0; i < s_shNum; i++)
        {
            GetFeature(coord, data_[i] + offsets_[level] * len,
                       glm::vec<InputDim, int>{ resolution_[level] },
                       offsets_[level + 1] - offsets_[level], len, tempFeature);
            for (int j = 0; j < len; j++)
            {
                tempFeature2[j] += tempFeature[j] * coeffs[i];
            }
        }

        for (int i = 0; i < len; i++)
            dst[i] = tempFeature2[i];
    }
#endif
};

template<typename T>
class Network;

template<typename T>
class NetworkGPUProxy
{
    T *weight_ = nullptr;
    T *bias_;
    int layerNum_;
    int inputDim_;
    int hiddenDim_;
    int outputDim_;

    friend Network<T>;

public:
    NetworkGPUProxy() = default;

#ifdef __CUDACC__
private:
    static __forceinline__ __device__ void MatVecMul(
        const T *__restrict__ inVec, const T *__restrict__ mat,
        T *__restrict__ outVec, int X, int Y)
    {
        int idxBase = 0;
        for (int i = 0; i < X; i++)
        {
            T temp = inVec[i];
            for (int j = 0; j < Y; j++)
                outVec[j] += temp * mat[idxBase + j];
            idxBase += Y;
        }
    }

    static __forceinline__ __device__ void MatVecMulAfterReLU(
        const T *__restrict__ inVec, const T *__restrict__ mat,
        T *__restrict__ outVec, int X, int Y)
    {
        int idxBase = 0;
        for (int i = 0; i < X; i++)
        {
            T temp = Maximum(inVec[i], (T)0.f);
            for (int j = 0; j < Y; j++)
                outVec[j] += temp * mat[idxBase + j];
            idxBase += Y;
        }
    }

    static __forceinline__ __device__ void ReLU6AndConcat(
        const T *__restrict__ inVec, T *__restrict__ outVec, int X)
    {
        for (int i = 0; i < X; i++)
        {
            outVec[i] = Minimum(Maximum(outVec[i], (T)0.f), (T)6.0f) + inVec[i];
        }
    }

public:
    __device__ auto PrintRepl() const noexcept
    {
        printf("Weight: %f %f %f %f\n", (float)weight_[0], (float)weight_[1],
               (float)weight_[2], (float)weight_[3]);
        if (bias_)
        {
            printf("Bias: %f %f %f %f\n", (float)bias_[0], (float)bias_[1],
                   (float)bias_[2], (float)bias_[3]);
        }
    }

    __device__ auto GetInputDim() const noexcept { return inputDim_; }
    __device__ auto GetOutputDim() const noexcept { return outputDim_; }

    __device__ void Inference(const T *input, T *output) const
    {
        assert(hiddenDim_ <= HIDDEN_FEATURE_DIM_UPPER_LIMIT);

        int offset1 = inputDim_ * hiddenDim_;
        auto weight = weight_;
        T hiddenBuffer[HIDDEN_FEATURE_DIM_UPPER_LIMIT]{};

        MatVecMul(input, weight, hiddenBuffer, inputDim_, hiddenDim_);
        if (bias_)
        {
            for (int i = 0; i < hiddenDim_; i++)
                hiddenBuffer[i] += bias_[i];
        }
        weight += offset1;

#if AGGRESIVE_OMIT_HIDDEN_LAYER == 0
        int offset2 = hiddenDim_ * hiddenDim_;
        T hiddenBuffer2[HIDDEN_FEATURE_DIM_UPPER_LIMIT]{};
        T *hiddenPtrs[2]{ hiddenBuffer, hiddenBuffer2 };
        for (int i = 1, inputIdx = 0; i < layerNum_; i++)
        {
            MatVecMulAfterReLU(hiddenPtrs[inputIdx], weight,
                               hiddenPtrs[1 - inputIdx], hiddenDim_,
                               hiddenDim_);
            weight += offset2, inputIdx = 1 - inputIdx;
        }
#else
        assert(layerNum_ == 1);
#endif
        MatVecMulAfterReLU(hiddenBuffer, weight, output, hiddenDim_,
                           outputDim_);
    }

    __device__ void InferenceWithConcat(const T *input, T *output) const
    {
        assert(inputDim_ == outputDim_);
        Inference(input, output);
        ReLU6AndConcat(input, output, outputDim_);
    }
#endif
};

struct NetworkGlobalPayload
{
    static constexpr inline int s_maxNetworkNum = 8;

    EncodingGPUProxy<NetworkFloat> baseEncoding, viewEncoding, lightEncoding,
        hgEncoding, albedoEncoding;

    SHEncodingGPUProxy<NetworkFloat, 2> shEncoding;
    NetworkGPUProxy<NetworkFloat> networks[s_maxNetworkNum];
    int networkNum;
    int viewEncDim;
};

struct RenderParametersPayload
{
    struct FrameBufferInfo
    {
        glm::u32vec2 size;
        float *bufferPtr;
        unsigned int channelNum;
    } frameBuffer;

    int randomOffset;
    float g;
    glm::vec3 albedo;

    EnvironmentMap::Proxy skybox;

    struct CameraInfo
    {
        glm::vec3 position;
        glm::vec3 forward, right, up;
    } camera;

    struct LightInfo
    {
        glm::vec3 dir, invDir;
        glm::ivec3 dirIsPos;
        glm::vec3 color;
    } light;

    struct TemporalInfo
    {
        CameraInfo camera;
        FrameBufferInfo frameBuffer;
        float *swapBuffer;
        float lerpCoeff;
    } lastFrameInfo;

    enum class ToneMapping
    {
        None,
        Gamma,
        ACES
    } toneMapping;

    float illumScale;
    bool excludeLightEncoding;
    bool fastDirectIllum;
    bool enableSkyboxBaking;
};

struct VolumePayload
{
    VolumeKernelData volume;
    TrMapKernelData trMap;
};

struct TCNNPayload
{
    void *handle;
    NetworkFloat *featureBuffer;
    half *radianceBuffer;
    std::size_t size;
    int featureDim;
};

struct GBufferPayload
{
    glm::vec4 *position;
    glm::vec3 *viewDir;
};