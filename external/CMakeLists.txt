# Isolate usage of mixing C++14/17 in case of incompatible ABIs;
# ExternalTCNN exposes interface with mere C fundamental types. 
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

FetchContent_Declare(
    tcnn
    GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
    GIT_TAG        3c7931caab7b691c6f30dffe20952ed727d935b9
    GIT_PROGRESS   TRUE
)

FetchContent_MakeAvailable(tcnn)

add_library(ExternalTCNN SHARED ExternalTCNN.cu)
set_target_properties(ExternalTCNN PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      CUDA_STANDARD 17)

target_compile_options(ExternalTCNN PUBLIC "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_link_libraries(ExternalTCNN PUBLIC tiny-cuda-nn fmt)
target_compile_definitions(ExternalTCNN PRIVATE KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/Network.kernel")
