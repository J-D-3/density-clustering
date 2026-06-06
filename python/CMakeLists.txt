# Optional pybind11 binding (OPTICS_BUILD_PYTHON=ON). Needs pybind11; point CMake
# at it with -Dpybind11_DIR=$(python -m pybind11 --cmakedir) or via CMAKE_PREFIX_PATH.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(optics_py optics_py.cpp)
target_link_libraries(optics_py PRIVATE optics::optics)
target_compile_features(optics_py PRIVATE cxx_std_20)
