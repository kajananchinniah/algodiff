include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

# Don't build off docs, tests, etc
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)

FetchContent_MakeAvailable(Eigen)
