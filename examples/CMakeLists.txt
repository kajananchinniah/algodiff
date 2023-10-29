cmake_minimum_required(VERSION 3.21)

project(
  algodiff_examples
  DESCRIPTION "Algodiff Examples"
  LANGUAGES CXX)

add_executable(algodiff_derivative_example derivative_example.cpp)
add_executable(algodiff_jacobian_example jacobian_example.cpp)

include(FetchContent)
FetchContent_Declare(
  algodiff
  GIT_REPOSITORY "https://github.com/kajananchinniah/algodiff/"
  GIT_TAG main)
FetchContent_MakeAvailable(algodiff)

target_link_libraries(algodiff_derivative_example algodiff)
target_link_libraries(algodiff_jacobian_example algodiff)
