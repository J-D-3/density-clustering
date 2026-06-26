cmake_minimum_required(VERSION 3.21)

project(density-clustering VERSION 1.0.0 LANGUAGES CXX)

# Optional Boost.Geometry rtree neighbor-search backend. OFF by default so the
# library builds with no external dependencies (only the vendored nanoflann).
option(OPTICS_ENABLE_BOOST_RTREE "Enable the optional Boost rtree backend" OFF)
option(OPTICS_ENABLE_HNSW "Enable the optional hnswlib HNSW approximate backend" OFF)
option(OPTICS_BUILD_TESTS "Build the OPTICS test suite" ${PROJECT_IS_TOP_LEVEL})
option(OPTICS_BUILD_EXAMPLES "Build the example programs" ${PROJECT_IS_TOP_LEVEL})
option(OPTICS_BUILD_PYTHON "Build the optional pybind11 Python binding" OFF)
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
    # CMake >= 3.30 removed the bundled FindBoost module; opting into CMP0167 NEW
    # consumes Boost's own BoostConfig (CONFIG mode) and silences the deprecation
    # warning. Guarded so older CMake (which lacks the policy) still configures.
    if(POLICY CMP0167)
        cmake_policy(SET CMP0167 NEW)
    endif()
    find_package(Boost REQUIRED)
    # Boost.Geometry is header-only; Boost::headers is provided by both BoostConfig
    # and the legacy FindBoost module, unlike the FindBoost-only Boost::boost.
    target_link_libraries(optics INTERFACE Boost::headers)
    target_compile_definitions(optics INTERFACE OPTICS_ENABLE_BOOST_RTREE)
endif()

if(OPTICS_ENABLE_HNSW)
    # hnswlib is vendored header-only under include/optics/hnswlib/ (already on the include
    # path), so there is no external package to find -- just flip on the gated backend.
    target_compile_definitions(optics INTERFACE OPTICS_ENABLE_HNSW)
endif()

if(OPTICS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(OPTICS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(OPTICS_BUILD_PYTHON)
    add_subdirectory(python)
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
