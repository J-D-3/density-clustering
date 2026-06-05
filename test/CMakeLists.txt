# Warnings on for all test targets. The visual test still verifies via assert(),
# so it also needs NDEBUG undefined in optimized configs.
set(OPTICS_WARN_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
set(OPTICS_ASSERT_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)

# Unit suite: doctest (TEST_CASE/CHECK; assertions independent of NDEBUG). The
# binary runs all cases and reports per-case (use ctest --output-on-failure to
# see doctest's report); registered once with CTest for cross-platform robustness.
add_executable(optics_tests test_main.cpp)
target_link_libraries(optics_tests PRIVATE optics::optics)
target_include_directories(optics_tests SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_compile_options(optics_tests PRIVATE ${OPTICS_WARN_OPTIONS})
add_test(NAME optics_tests COMMAND optics_tests)

# Visual / export test: clusters synthetic clouds, asserts structure, and emits
# CSVs for tools/visualize.py.
add_executable(optics_visual_test ClusterImage_Test/test_ClusterImages.cpp)
target_link_libraries(optics_visual_test PRIVATE optics::optics)
target_compile_options(optics_visual_test PRIVATE ${OPTICS_WARN_OPTIONS} ${OPTICS_ASSERT_OPTIONS})
add_test(NAME optics_visual_test COMMAND optics_visual_test)

# Performance benchmark (build in a Release config for meaningful numbers).
# Not registered as a ctest.
add_executable(optics_benchmark Benchmark/benchmark.cpp)
target_link_libraries(optics_benchmark PRIVATE optics::optics)
target_compile_options(optics_benchmark PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)

# Performance-regression harness (nanobench). Release config; emits optics_perf.csv.
# Not a ctest (timings vary by machine).
add_executable(optics_perf Benchmark/perf.cpp)
target_link_libraries(optics_perf PRIVATE optics::optics)
target_include_directories(optics_perf SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# Large-scale (1e6-1e7) 3D timing harness. Release config; not a ctest.
add_executable(optics_scale Benchmark/scale.cpp)
target_link_libraries(optics_scale PRIVATE optics::optics)
