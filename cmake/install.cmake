set(package algodiff)

include(GNUInstallDirs)
install(
  TARGETS algodiff
  EXPORT algodiffTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/executables/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN "*/unsupported**" EXCLUDE)

include(CMakePackageConfigHelpers)

# generate the config file that is includes the exports
configure_package_config_file(
  cmake/algodiffConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/algodiffConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
  NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# generate the version file for the config file
write_basic_package_version_file(
  "algodiffConfigVersion.cmake"
  VERSION "${algodiff_VERSION_MAJOR}.${algodiff_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion)

# install the configuration file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/algodiffConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/algodiffConfigVersion.cmake
        DESTINATION share/algodiff/cmake)

install(
  EXPORT algodiffTargets
  NAMESPACE algodiff::
  DESTINATION share/algodiff/cmake)
