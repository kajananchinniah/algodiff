cmake_minimum_required(VERSION 3.21)

project(test_algodiff LANGUAGES CXX)

# Ignore clang-tidy
if(CMAKE_CXX_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY_ALGODIFF_TEST_TMP ${CMAKE_CXX_CLANG_TIDY})
  set(CMAKE_CXX_CLANG_TIDY "")
endif()

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.0.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

include(CTest)
include(Catch)

add_executable(dual_number_test src/dual_number_test.cpp)
target_link_libraries(dual_number_test PRIVATE algodiff
                                               Catch2::Catch2WithMain)
target_compile_features(dual_number_test PRIVATE cxx_std_17)

catch_discover_tests(dual_number_test)

# Restore clang-tidy
if(CMAKE_CXX_CLANG_TIDY_ALGODIFF_TEST_TMP)
  set(CMAKE_CXX_CLANG_TIDY ${CMAKE_CXX_CLANG_TIDY_ALGODIFF_TEST_TMP})
endif()
