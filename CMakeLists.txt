cmake_minimum_required(VERSION 3.21)

project(OPTICS-Clustering VERSION 0.9.0 LANGUAGES CXX)

# Optional Boost.Geometry rtree neighbor-search backend. OFF by default so the
# library builds with no external dependencies (only the vendored nanoflann).
option(OPTICS_ENABLE_BOOST_RTREE "Enable the optional Boost rtree backend" OFF)
option(OPTICS_BUILD_TESTS "Build the OPTICS test suite" ${PROJECT_IS_TOP_LEVEL})
option(OPTICS_BUILD_EXAMPLES "Build the example programs" ${PROJECT_IS_TOP_LEVEL})
option(OPTICS_INSTALL "Generate install/package rules" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)  # defines CMAKE_INSTALL_INCLUDEDIR (used below) before first use
find_package(Threads REQUIRED)

# Header-only library, consumed as an INTERFACE target.
add_library(optics INTERFACE)
add_library(optics::optics ALIAS optics)
target_include_directories(optics INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(optics INTERFACE cxx_std_20)
target_link_libraries(optics INTERFACE Threads::Threads)

if(OPTICS_ENABLE_BOOST_RTREE)
    find_package(Boost REQUIRED)
    target_link_libraries(optics INTERFACE Boost::boost)
    target_compile_definitions(optics INTERFACE OPTICS_ENABLE_BOOST_RTREE)
endif()

if(OPTICS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(OPTICS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optional API documentation. `cmake --build <dir> --target docs` when Doxygen is
# installed; a no-op target otherwise so the call never fails.
if(PROJECT_IS_TOP_LEVEL)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    endif()
endif()

# Install + package config so downstream can `find_package(optics)` or FetchContent.
if(OPTICS_INSTALL)
    include(CMakePackageConfigHelpers)

    set(OPTICS_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/optics")

    install(TARGETS optics EXPORT opticsTargets)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/optics"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(EXPORT opticsTargets
            FILE opticsTargets.cmake
            NAMESPACE optics::
            DESTINATION "${OPTICS_CMAKE_DIR}")

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/opticsConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/opticsConfig.cmake"
        INSTALL_DESTINATION "${OPTICS_CMAKE_DIR}")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/opticsConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/opticsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/opticsConfigVersion.cmake"
        DESTINATION "${OPTICS_CMAKE_DIR}")
endif()
