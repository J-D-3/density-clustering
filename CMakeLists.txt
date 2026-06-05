cmake_minimum_required(VERSION 3.20)

project(OPTICS-Clustering LANGUAGES CXX)

# Optional Boost.Geometry rtree neighbor-search backend. OFF by default so the
# library builds with no external dependencies (only the vendored nanoflann).
option(OPTICS_ENABLE_BOOST_RTREE "Enable the optional Boost rtree backend" OFF)
option(OPTICS_BUILD_TESTS "Build the OPTICS test suite" ON)

find_package(Threads REQUIRED)

# Header-only library, consumed as an INTERFACE target.
add_library(optics INTERFACE)
add_library(optics::optics ALIAS optics)
target_include_directories(optics INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
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
