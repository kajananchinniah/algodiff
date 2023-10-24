include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

# Don't build off docs, tests, etc
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)

FetchContent_MakeAvailable(Eigen)
