cmake_minimum_required(VERSION 3.17)
project(MRBNN LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit)

include(FetchContent)

# Enable UTF-8 and warnings.
if(MSVC)
    set(GlobalCXXFlags /utf-8 /W3 /wd4996)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${GlobalCXXFlags}>")
    foreach(flag IN LISTS GlobalCXXFlags)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
    endforeach()
else()
    set(GlobalCXXFlags -fexec-charset=UTF-8 -finput-charset=UTF-8 -Wall)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${GlobalCXXFlags}>")
    foreach(flag IN LISTS GlobalCXXFlags)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
    endforeach()
endif()

set(GlobalCUDAFlags "--expt-relaxed-constexpr" "--extended-lambda" "--diag-suppress" "20012")
foreach(flag IN LISTS GlobalCUDAFlags)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${flag}>")
endforeach()

add_subdirectory(external)
add_subdirectory(src)