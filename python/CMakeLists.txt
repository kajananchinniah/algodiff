cmake_minimum_required(VERSION 3.22)

project(
  algodiff_python
  DESCRIPTION "Algodiff Python binding"
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.11.1)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

FetchContent_Declare(
  algodiff
  GIT_REPOSITORY "https://github.com/kajananchinniah/algodiff/"
  GIT_TAG main)
FetchContent_MakeAvailable(algodiff)

pybind11_add_module(algodiff_py src/python_bindings.cpp)
target_link_libraries(algodiff_py PRIVATE algodiff)
