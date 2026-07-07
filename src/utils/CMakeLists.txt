add_library(stb_image STATIC stbi/stb_image.cpp)

add_library(Utils STATIC Utils.cpp)
target_link_libraries(Utils PUBLIC CUDA::cudart glm stb_image)
# Not-that-good practice, just to make src/ always part of include path.
target_include_directories(Utils PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/..)