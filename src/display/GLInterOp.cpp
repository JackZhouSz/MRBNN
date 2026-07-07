/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "glad/gl.h"

#include "stbi/stb_image.h"
#include "stbi/stb_image_write.h"

#include "GLInterOp.hpp"

#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static std::string_view GetGLErrorString(GLenum error)
{
    switch (error)
    {
    case GL_NO_ERROR:
        return "No error";
    case GL_INVALID_ENUM:
        return "Invalid enum";
    case GL_INVALID_VALUE:
        return "Invalid value";
    case GL_INVALID_OPERATION:
        return "Invalid operation";
    case GL_STACK_OVERFLOW:
        return "Stack overflow";
    case GL_STACK_UNDERFLOW:
        return "Stack underflow";
    case GL_OUT_OF_MEMORY:
        return "Out of memory";
    default:
        return "Unknown GL error";
    }
}

static void CheckGLError()
{
    GLenum error = glGetError();
    CheckError(error == GL_NO_ERROR,
               LAZY_STR(std::format("GL error: Error code {} - {}", error,
                                    GetGLErrorString(error))));
}

#ifndef GL_DISPLAY_USE_BLIT

template<typename T, typename U, typename V>
void CheckCompileOrLinkError(GLuint id, T statusGetter, U maxLenGetter,
                             V logGetter)
{
    GLint success = 0;
    statusGetter(id, &success);
    if (success == GL_TRUE)
        return;

    // Else compile or link fails.
    GLint maxLen = 0;
    maxLenGetter(id, &maxLen);
    std::string errorLog;
    errorLog.resize(maxLen);

    auto logPtr = errorLog.data();
    int realLen = 0;
    logGetter(id, maxLen, &realLen, logPtr);

    CheckError(false, std::string_view{ logPtr, (unsigned int)realLen });
}

class GLShader
{
public:
    GLShader(const char *contentPtr, const GLenum type)
        : shaderID_{ glCreateShader(type) }
    {
        auto newShader = shaderID_.Get();
        glShaderSource(newShader, 1, &contentPtr, NULL);
        glCompileShader(newShader);

        CheckCompileOrLinkError(
            newShader,
            [](auto id, auto *successPtr) {
                glGetShaderiv(id, GL_COMPILE_STATUS, successPtr);
            },
            [](auto id, auto *lengthPtr) {
                glGetShaderiv(id, GL_INFO_LOG_LENGTH, lengthPtr);
            },
            [](auto id, auto maxLen, auto *lengthPtr, auto *logPtr) {
                glGetShaderInfoLog(id, (GLint)maxLen - 1, lengthPtr, logPtr);
            });
    }
    auto Get() const noexcept { return shaderID_.Get(); }

private:
    DEFINE_SINGLE_DELETER_GO_AROUND_CPP17(Deleter, glDeleteShader);

    UniqueResource<GLuint, Deleter> shaderID_;
};

class GLProgram
{
public:
    template<typename... Args>
    GLProgram(Args... shaders) : programID_{ glCreateProgram() }
    {
        auto newShaderAssembly = programID_.Get();
        (glAttachShader(newShaderAssembly, shaders), ...);
        glLinkProgram(newShaderAssembly);

        CheckCompileOrLinkError(
            newShaderAssembly,
            [](auto id, auto *successPtr) {
                glGetProgramiv(id, GL_LINK_STATUS, successPtr);
            },
            [](auto id, auto *lengthPtr) {
                glGetProgramiv(id, GL_INFO_LOG_LENGTH, lengthPtr);
            },
            [](auto id, auto maxLen, auto *lengthPtr, auto *logPtr) {
                glGetProgramInfoLog(id, (GLint)maxLen - 1, lengthPtr, logPtr);
            });
    }
    auto GetUniformLocation(const char *name) const noexcept
    {
        return glGetUniformLocation(programID_.Get(), name);
    }
    void Activate() const noexcept { glUseProgram(programID_.Get()); }

private:
    DEFINE_SINGLE_DELETER_GO_AROUND_CPP17(Deleter, glDeleteProgram);

    UniqueResource<GLuint, Deleter> programID_;
};

class Quad
{
    inline static constexpr GLfloat s_quadVertices_[] = {
        -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f,  0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
    };

public:
    Quad()
    {
        GLuint VAO = 0, VBO = 0;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        vao_.Reloc(VAO), vbo_.Reloc(VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(s_quadVertices_), s_quadVertices_,
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGLError();
    }

    void Draw() const noexcept
    {
        constexpr int triangleNum = 6;
        static_assert(std::size(s_quadVertices_) == triangleNum * 3);

        glBindVertexArray(vao_.Get());
        glDrawArrays(GL_TRIANGLES, 0, triangleNum);
        glBindVertexArray(0);
    }

private:
    DEFINE_DELETER_GO_AROUND_CPP17(VAODeleter, glDeleteVertexArrays);
    DEFINE_DELETER_GO_AROUND_CPP17(VBODeleter, glDeleteBuffers);

    UniqueResource<GLuint, VAODeleter> vao_;
    UniqueResource<GLuint, VBODeleter> vbo_;
};

class CanvasSingleton
{
    inline static constexpr const char *vertexShaderSource_ =
        R"(#version 430 core
    layout(location = 0) in vec3 aPos;
    out vec2 TexCoords;
    void main()
    {
        TexCoords = (aPos.xy + vec2(1, 1)) / 2;
        gl_Position = vec4(aPos, 1.0);
        return;
    })";

    inline static constexpr const char *fragmentShaderSource_ =
        R"(#version 430 core
    out vec4 FragColor;
    in vec2 TexCoords;
    uniform sampler2D canvasTexture;
    void main()
    {
        FragColor = texture(canvasTexture, TexCoords);
        return;
    })";

    static GLProgram GenerateProgram_()
    {
        std::string fragSrcCache;
        const char *fragSrcPtr = nullptr;

        try
        {
            auto fragPath =
                ReadAllFromFile(
                    R"(D:\Work\Graphics\Volume Rendering\Tests\NeuVolEvalCUDA\shaders\config.txt)")
                    .str();
            fragSrcCache = ReadAllFromFile(fragPath).str();
            fragSrcPtr = fragSrcCache.c_str();
        }
        catch (...)
        {
            std::cerr << "Cannot load fragment shader correctly, use default "
                         "version\n";
            fragSrcPtr = fragmentShaderSource_;
        }

        GLShader vertexShader{ vertexShaderSource_, GL_VERTEX_SHADER },
            fragmentShader{ fragSrcPtr, GL_FRAGMENT_SHADER };

        GLProgram program{ vertexShader.Get(), fragmentShader.Get() };
        return program;
    }

    // Here name "canvasTexture" is same as texture in fragment shader.
    CanvasSingleton()
        : program_{ GenerateProgram_() },
          textureLoc_{ program_.GetUniformLocation("canvasTexture") }
    {
    }

public:
    static CanvasSingleton &GetCanvas()
    {
        static CanvasSingleton canvas;
        return canvas;
    }

    void Draw(int textureID) const
    {
        program_.Activate();
        glUniform1i(textureLoc_, textureID);
        quad_.Draw();
    }

private:
    GLProgram program_;
    Quad quad_;
    GLint textureLoc_;
};

#endif

static std::size_t GetElemSize(GLDisplayBase::PixelFormat format)
{
    using PixelFormat = GLDisplayBase::PixelFormat;
    if (format == PixelFormat::Float4)
        return sizeof(float) * 4;
    else if (format == PixelFormat::Float3)
        return sizeof(float) * 3;
    else if (format == PixelFormat::UnsignedChar4)
        return sizeof(unsigned char) * 4;
    else if (format == PixelFormat::Float)
        return sizeof(float);
    throw std::runtime_error{ "Unknown pixel format" };
}

struct GLFormat
{
    unsigned int gpuPixelFormat;
    unsigned int cpuPixelFormat;
    unsigned int rawDataType;
};

static GLFormat GetGLFormat(GLDisplayBase::PixelFormat format)
{
    using PixelFormat = GLDisplayBase::PixelFormat;
    if (format == PixelFormat::Float4)
        return { GL_RGBA32F, GL_RGBA, GL_FLOAT };
    else if (format == PixelFormat::Float3)
        return { GL_RGB32F, GL_RGB, GL_FLOAT };
    else if (format == PixelFormat::UnsignedChar4)
        return { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE };
    else if (format == PixelFormat::Float)
        return { GL_R32F, GL_RED, GL_FLOAT };
    throw std::runtime_error{ "Unknown pixel format" };
}

GLDisplayBase::GLDisplayBase(PixelFormat format) : format_{ format }
{
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    pbo_.Reloc(pbo);

    GLuint texture;
    glGenTextures(1, &texture);
    texture_.Reloc(texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

#ifdef GL_DISPLAY_USE_BLIT
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    fbo_.Reset(fbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif

    CheckGLError();
}

void GLDisplayBase::NormalizeInputWidthAndHeight_(
    unsigned int &newWidth, unsigned int &newHeight) const noexcept
{
    if (newWidth == 0)
        newWidth = width_;
    if (newHeight == 0)
        newHeight = height_;
}

void GLDisplayBase::ResizePBO_(std::size_t newSize)
{
    glBindBuffer(GL_ARRAY_BUFFER, pbo_.Get());
    glBufferData(GL_ARRAY_BUFFER, newSize * GetElemSize(format_), nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CheckGLError();
    pboSize_ = newSize;
}

void GLDisplayBase::ResizeTexture_(unsigned int newWidth,
                                   unsigned int newHeight)
{
    auto glFormat = GetGLFormat(format_);
    // Allocate an texture, without writing anything.
    glBindTexture(GL_TEXTURE_2D, texture_.Get());
    glTexImage2D(GL_TEXTURE_2D, 0, glFormat.gpuPixelFormat, newWidth, newHeight,
                 0, glFormat.cpuPixelFormat, glFormat.rawDataType, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

#ifdef GL_DISPLAY_USE_BLIT
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_.Get());
    auto err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    CheckError(
        err == GL_FRAMEBUFFER_COMPLETE,
        LAZY_STR(fmt::format(
            "Cannot generate complete frame buffer, error code: {}", err)));
#endif

    CheckGLError();
    width_ = newWidth;
    height_ = newHeight;
}

void GLDisplayBase::SaveHDR(const std::filesystem::path &path, void *src) const
{
    stbi_flip_vertically_on_write(true);
    auto localePath = path.string();
    if (format_ == PixelFormat::Float4)
    {
        stbi_write_hdr(localePath.c_str(), width_, height_, 4, (float *)src);
    }
    else if (format_ == PixelFormat::Float3)
    {
        stbi_write_hdr(localePath.c_str(), width_, height_, 3, (float *)src);
    }
    else if (format_ == PixelFormat::UnsignedChar4)
    {
        stbi_write_bmp(localePath.c_str(), width_, height_, 4, src);
    }
    else if (format_ == PixelFormat::Float)
    {
        stbi_write_hdr(localePath.c_str(), width_, height_, 1, (float *)src);
    }
    else
        throw std::runtime_error{ "Unsupported pixel format" };

    return;
}

bool GLDisplayBase::TryResize(unsigned int width, unsigned int height,
                              std::size_t pboSize)
{
    NormalizeInputWidthAndHeight_(width, height);

    bool isTextureSameSize = width == width_ && height == height_;
    if (!isTextureSameSize)
    {
        ResizeTexture_(width, height);
    }

    std::size_t textureSize = (std::size_t)width_ * height_;
    if (pboSize == 0)
        pboSize = textureSize;
    else if (pboSize < textureSize)
        throw std::runtime_error{ "PBO size is too small for a texture" };

    if (pboSize != pboSize_)
    {
        ResizePBO_(pboSize);
        return true;
    }

    return false;
}

void GLDisplayBase::TransferFromPBOToTexture_() const
{
    glActiveTexture(GL_TEXTURE0 + s_texUnitID_);
    glBindTexture(GL_TEXTURE_2D, texture_.Get());
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_.Get());
    auto glFormat = GetGLFormat(format_);
    glTexImage2D(GL_TEXTURE_2D, 0, glFormat.gpuPixelFormat, width_, height_, 0,
                 glFormat.cpuPixelFormat, glFormat.rawDataType, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GLDisplayBase::Draw(unsigned int width, unsigned int height) const
{
    NormalizeInputWidthAndHeight_(width, height);
    TransferFromPBOToTexture_();

#ifdef GL_DISPLAY_USE_BLIT
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_.Get());
    glBlitFramebuffer(0, 0, width_, height_, 0, 0, width, height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
#else
    auto &canvas = CanvasSingleton::GetCanvas();
    glViewport(0, 0, width, height);
    canvas.Draw(s_texUnitID_);
#endif
    CheckGLError();
}

GLInterOpDisplay::GLInterOpDisplay(PixelFormat format) : GLDisplayBase{ format }
{
    int isDisplayDevice;
    CheckCUDAError(cudaDeviceGetAttribute(&isDisplayDevice,
                                          cudaDevAttrKernelExecTimeout, 0));
    CheckError(isDisplayDevice, "InterOp memory must be on display device.");
}

void GLInterOpDisplay::TryResize(unsigned int width, unsigned int height,
                                 std::size_t pboSize)
{
    if (!GLDisplayBase::TryResize(width, height, pboSize))
        return;

    cudaGraphicsResource_t newCudaRes{};
    CheckCUDAError(cudaGraphicsGLRegisterBuffer(&newCudaRes, GetPBO(),
                                                cudaGraphicsMapFlagsNone));

    cudaRes_.Reset(newCudaRes);
}

void GLInterOpDisplay::SaveToFile(void *devicePtr,
                                  const std::filesystem::path &path)
{
    if (devicePtr == nullptr)
        return;

    auto size = GetElemSize(GetPixelFormat()) * GetPBOSize();
    auto hostBuffer = std::make_unique<std::byte[]>(size);
    auto rawHostPtr = hostBuffer.get();
    CheckCUDAError(
        cudaMemcpy(rawHostPtr, devicePtr, size, cudaMemcpyDeviceToHost));
    SaveHDR(path, rawHostPtr);
}