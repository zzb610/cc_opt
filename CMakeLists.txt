# Minimum CMake required
cmake_minimum_required(VERSION 3.14.0)

project(CTEST_CUSTOM_POST_TEST C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-std=c++14)
add_compile_options(-fPIC)
add_compile_options(-Wl,--no-as-needed)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) 

set(CMAKE_BUILD_TYPE Debug)

include_directories(utils)

add_subdirectory(ga)
add_subdirectory(pso)
add_subdirectory(test)
add_subdirectory(utils)