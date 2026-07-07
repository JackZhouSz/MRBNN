FetchContent_Declare(
  glfw
  GIT_REPOSITORY    https://github.com/glfw/glfw.git
  GIT_TAG           7b6aead9fb88b3623e3b3725ebb42670cbe4c579
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY    https://github.com/ocornut/imgui.git
  GIT_TAG           9a5d5c45f54b1301ea471622eddede70384243af
) # imgui 1.92.4

FetchContent_MakeAvailable(imgui)

add_library(imgui_glfw STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp

  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_link_libraries(imgui_glfw PUBLIC glfw)
target_include_directories(imgui_glfw PUBLIC ${imgui_SOURCE_DIR} 
                                             ${imgui_SOURCE_DIR}/backends)

# ** For users' convenience, we provide glad2 in the released directories
# ** directly to avoid requirement of Python. If you want to generate glad
# ** files yourself, uncomment the commented glad-related code (and comment
# ** out current glad-related code) below. You need to build glad target
# ** first to generate these files, and then build glad library.

# FetchContent_Declare(
#   glad
#   GIT_REPOSITORY    https://github.com/Dav1dde/glad.git
#   GIT_TAG           73db193f853e2ee079bf3ca8a64aa2eaf6459043
# ) # glad 2.0.8

# FetchContent_MakeAvailable(glad)

# find_package(Python COMPONENTS Interpreter)
# set(GLAD_DIR ${glad_SOURCE_DIR})
# add_subdirectory(${glad_SOURCE_DIR}/cmake ${glad_BINARY_DIR})
# glad_add_library(glad STATIC REPRODUCIBLE API gl:core=4.3)
# set_target_properties(glad PROPERTIES LANGUAGE CXX)

add_library(glad STATIC glad/src/gl.c)
set_target_properties(glad PROPERTIES LANGUAGE CXX)
target_include_directories(glad PUBLIC glad/include)

add_library(WindowLib STATIC Window.cpp GLInterOp.cpp FileDialog.cpp)
target_link_libraries(WindowLib PUBLIC glfw glad Utils imgui_glfw)