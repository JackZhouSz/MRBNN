FetchContent_Declare(
  glm
  GIT_REPOSITORY    https://github.com/g-truc/glm.git
  GIT_TAG           0af55ccecd98d4e5a8d1fad7de25ba429d60e863
) # 1.0.1
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
  json
  GIT_REPOSITORY    https://github.com/nlohmann/json.git
  GIT_TAG           55f93686c01528224f448c19128836e7df245f72
) # 3.12.0
FetchContent_MakeAvailable(json)
# Astyle-wheel is GPL license and the company asks me to remove it,
# even though it's not even part of nlohmann/json C++ code.
file(REMOVE_RECURSE ${json_SOURCE_DIR}/tools)

add_subdirectory(utils)
add_subdirectory(display)
add_subdirectory(render)

add_subdirectory(test)
add_subdirectory(tools)

add_executable(main main.cpp gui/GUI.cpp gui/ConfigLoader.cpp)
target_link_libraries(main PRIVATE WindowLib RenderLib)
add_custom_command(
  TARGET main POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:ExternalTCNN>
          $<TARGET_FILE_DIR:main>)