add_executable(display.test display.test.cpp)
target_link_libraries(display.test PRIVATE WindowLib)

add_executable(mipmap.test mipmap.test.cu)
target_link_libraries(mipmap.test PRIVATE RenderLib)

add_executable(network.test network.test.cu)
target_link_libraries(network.test PRIVATE RenderLib)

add_executable(trmap.test trmap.test.cu)
target_link_libraries(trmap.test PRIVATE RenderLib)

add_executable(envmap.test envmap.test.cu)
target_link_libraries(envmap.test PRIVATE RenderLib)