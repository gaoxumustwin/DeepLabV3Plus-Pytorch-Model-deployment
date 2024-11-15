project(NCNN_DEMO)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_BUILD_TYPE Debug)


set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/mnt/e/ubuntu_20.04/ncnn/build/install")
find_package(ncnn   REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(deeplabv3plus src/deeplabv3plus.cpp)
target_link_libraries(deeplabv3plus ncnn ${OpenCV_LIBS})