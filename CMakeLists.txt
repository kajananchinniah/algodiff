cmake_minimum_required(VERSION 3.22)

project(
  algodiff
  VERSION 0.0.1
  LANGUAGES CXX)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_DOCS "Build docs" ON)
option(STATIC_ANALYSIS "Static analysis" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/ClangTidy.cmake)
include(docs/Doxygen.cmake)

include_directories(include)

add_subdirectory(external)

add_library(
  algodiff SHARED src/algodiff.cpp src/dual_number.cpp src/dual_number_ops.cpp
                  src/dual_number_eigen.cpp src/forward_mode.cpp)
target_link_libraries(algodiff PUBLIC Eigen3::Eigen)

target_include_directories(
  algodiff PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                  $<INSTALL_INTERFACE:include>)

include(cmake/install.cmake)

include(CTest)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
