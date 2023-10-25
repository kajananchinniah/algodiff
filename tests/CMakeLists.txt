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
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(Catch)

add_executable(dual_number_test src/dual_number_test.cpp)
target_link_libraries(dual_number_test PRIVATE algodiff Catch2::Catch2WithMain)
target_compile_features(dual_number_test PRIVATE cxx_std_17)

catch_discover_tests(dual_number_test)

add_executable(forward_mode_function_test src/forward_mode_function_test.cpp)
target_link_libraries(forward_mode_function_test PRIVATE algodiff
                                                         Catch2::Catch2WithMain)
target_compile_features(forward_mode_function_test PRIVATE cxx_std_17)

catch_discover_tests(forward_mode_function_test)

add_executable(forward_mode_derivative_test
               src/forward_mode_derivative_test.cpp)
target_link_libraries(forward_mode_derivative_test
                      PRIVATE algodiff Catch2::Catch2WithMain)
target_compile_features(forward_mode_derivative_test PRIVATE cxx_std_17)

catch_discover_tests(forward_mode_derivative_test)

add_executable(forward_mode_multidimensional_derivative_test
               src/forward_mode_multidimensional_derivative_test.cpp)
target_link_libraries(forward_mode_multidimensional_derivative_test
                      PRIVATE algodiff Catch2::Catch2WithMain)
target_compile_features(forward_mode_multidimensional_derivative_test
                        PRIVATE cxx_std_17)

catch_discover_tests(forward_mode_multidimensional_derivative_test)

# Restore clang-tidy
if(CMAKE_CXX_CLANG_TIDY_ALGODIFF_TEST_TMP)
  set(CMAKE_CXX_CLANG_TIDY ${CMAKE_CXX_CLANG_TIDY_ALGODIFF_TEST_TMP})
endif()
