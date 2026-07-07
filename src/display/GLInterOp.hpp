/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#pragma once
// glad MUST be included before cuda_gl_interop.h
#include "glad/gl.h"

#include "cuda_gl_interop.h"

#include "Utils.hpp"

#include <filesystem>

#define DEFINE_DELETER_GO_AROUND_CPP17(Name, Func) \
    struct Name                                    \
    {                                              \
        template<typename T>                       \
        void operator()(T o) const                 \
        {                                          \
            Func(1, &o);                           \
        }                                          \
    }

#define DEFINE_SINGLE_DELETER_GO_AROUND_CPP17(Name, Func) \
    struct Name                                           \
    {                                                     \
        template<typename T>                              \
        void operator()(T o) const                        \
        {                                                 \
            Func(o);                                      \
        }                                                 \
    }

// #define GL_DISPLAY_USE_BLIT

class GLDisplayBase
{
public:
    enum class PixelFormat
    {
        Float,
        UnsignedChar4,
        Float3,
        Float4
    };

    void Draw(unsigned int width = 0, unsigned int height = 0) const;
    auto GetWidth() const noexcept { return width_; }
    auto GetHeight() const noexcept { return height_; }
    auto GetPBOSize() const noexcept { return pboSize_; }
    auto GetPixelFormat() const noexcept { return format_; }

protected: // Not usable for normal users.
    auto GetPBO() const noexcept { return pbo_.Get(); }
    void SaveHDR(const std::filesystem::path &path, void *src) const;
    bool TryResize(unsigned int width, unsigned int height,
                   std::size_t pboSize = 0);

    GLDisplayBase(PixelFormat format);
    GLDisplayBase(const GLDisplayBase &another) = delete;
    GLDisplayBase &operator=(const GLDisplayBase &another) = delete;
    GLDisplayBase(GLDisplayBase &&another) noexcept = default;
    GLDisplayBase &operator=(GLDisplayBase &&another) noexcept = default;
    ~GLDisplayBase() = default;

private:
    static inline constexpr int s_texUnitID_ = 0;

    DEFINE_DELETER_GO_AROUND_CPP17(TextureDeleter, glDeleteTextures);
    DEFINE_DELETER_GO_AROUND_CPP17(PBODeleter, glDeleteBuffers);

    void NormalizeInputWidthAndHeight_(unsigned int &newWidth,
                                       unsigned int &newHeight) const noexcept;

    void ResizePBO_(std::size_t newSize);
    void ResizeTexture_(unsigned int newWidth, unsigned int newHeight);
    void TransferFromPBOToTexture_() const;

    PixelFormat format_;

    UniqueResource<GLuint, PBODeleter> pbo_;
    UniqueResource<GLuint, TextureDeleter> texture_;
    unsigned int width_ = 0, height_ = 0;
    std::size_t pboSize_ = 0;

#ifdef GL_DISPLAY_USE_BLIT
    DEFINE_DELETER_GO_AROUND_CPP17(FBODeleter, glDeleteFramebuffers);
    void ResetFBO_();
    UniqueResource<GLuint, FBODeleter> fbo_;
#endif
};

// In its lifetime (i.e. the device pointer is still not unmapped), the related
// PBO should NOT be accessed by graphics API. Thus, it's normally re-generated
// every new frame; see details at:
// https://stackoverflow.com/questions/6481123/cuda-and-opengl-interop, and
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html.
class InterPtrGuard
{
public:
    InterPtrGuard() : srcResource_{}, devicePtr_{} {}
    InterPtrGuard(cudaGraphicsResource_t res)
    {
        CheckCUDAError(cudaGraphicsMapResources(1, &res, 0));
        srcResource_.Reloc(res);
        CheckCUDAError(
            cudaGraphicsResourceGetMappedPointer(&devicePtr_, nullptr, res));
    }
    InterPtrGuard(const InterPtrGuard &) = delete;
    InterPtrGuard &operator=(const InterPtrGuard &) = delete;
    InterPtrGuard(InterPtrGuard &&another) noexcept
        : srcResource_{ std::move(another.srcResource_) },
          devicePtr_{ std::exchange(another.devicePtr_, nullptr) }
    {
    }
    InterPtrGuard &operator=(InterPtrGuard &&another) noexcept
    {
        srcResource_ = std::move(another.srcResource_);
        devicePtr_ = std::exchange(another.devicePtr_, nullptr);
    }
    ~InterPtrGuard() = default;

    auto GetDevicePtr() const noexcept { return devicePtr_; }

private:
    struct ResourceUnmapper
    {
        void operator()(cudaGraphicsResource_t res) const
        {
            cudaGraphicsUnmapResources(1, &res, 0);
        }
    };

    using CUDAGraphicsResourceMapper =
        UniqueResource<cudaGraphicsResource_t, ResourceUnmapper,
                       cudaGraphicsResource_t{}, true>;

    CUDAGraphicsResourceMapper srcResource_;
    void *devicePtr_;
};

class GLInterOpDisplay : public GLDisplayBase
{
public:
    // By default rgba.
    GLInterOpDisplay(PixelFormat format = PixelFormat::Float4);

    GLInterOpDisplay(const GLInterOpDisplay &another) = delete;
    GLInterOpDisplay &operator=(const GLInterOpDisplay &another) = delete;
    GLInterOpDisplay(GLInterOpDisplay &&another) noexcept = default;
    GLInterOpDisplay &operator=(GLInterOpDisplay &&another) noexcept = default;
    ~GLInterOpDisplay() = default;

    void TryResize(unsigned int width, unsigned int height,
                   std::size_t pboSize = 0);
    void SaveToFile(void *devicePtr, const std::filesystem::path &path);
    auto GetDevicePtrGuard() const noexcept
    {
        return InterPtrGuard{ cudaRes_.Get() };
    }

private:
    using ResourceDeleter = Deleter<cudaGraphicsUnregisterResource>;

    using CUDAGraphicsResource =
        UniqueResource<cudaGraphicsResource_t, ResourceDeleter,
                       cudaGraphicsResource_t{}, true>;

    CUDAGraphicsResource cudaRes_;
};
