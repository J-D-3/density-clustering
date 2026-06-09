# Builds the native `_optics` pybind11 extension. Works two ways:
#
#  * standalone (pip / scikit-build-core): a top-level project that uses the header-only
#    library from ../include and installs the module into the `optics` wheel package.
#  * in-tree (configured from the parent with -DOPTICS_BUILD_PYTHON=ON): the parent already
#    defined the optics::optics target, so we just link it and skip the wheel install.
#
# Point CMake at pybind11 with -Dpybind11_DIR=$(python -m pybind11 --cmakedir) or via
# CMAKE_PREFIX_PATH (scikit-build-core does this automatically from the build requirements).

cmake_minimum_required(VERSION 3.21)

if(NOT TARGET optics::optics)
    # Standalone build: synthesize a header-only interface target for ../include.
    project(optics_python LANGUAGES CXX)
    add_library(optics_headers INTERFACE)
    target_include_directories(optics_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_compile_features(optics_headers INTERFACE cxx_std_20)
    add_library(optics::optics ALIAS optics_headers)
    set(_optics_standalone TRUE)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_optics optics_py.cpp)
target_link_libraries(_optics PRIVATE optics::optics)
target_compile_features(_optics PRIVATE cxx_std_20)

if(_optics_standalone)
    # Install the extension into the `optics` package directory of the wheel.
    install(TARGETS _optics LIBRARY DESTINATION optics COMPONENT python)
endif()
