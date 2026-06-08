# Warnings on for all test targets. The visual test still verifies via assert(),
# so it also needs NDEBUG undefined in optimized configs.
option(OPTICS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors in the test build" OFF)
set(OPTICS_WARN_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
if(OPTICS_WARNINGS_AS_ERRORS)
    list(APPEND OPTICS_WARN_OPTIONS
        $<$<CXX_COMPILER_ID:MSVC>:/WX>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Werror>)
endif()
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

# Per-backend timing over CSV clouds (paired with tools/timing_compare.py, which adds
# scikit-learn OPTICS). Release config; not a ctest.
add_executable(optics_backend_compare Benchmark/backend_compare.cpp)
target_link_libraries(optics_backend_compare PRIVATE optics::optics)

# Approximate-backend probe: recall vs eps level and dimension (explains when the
# approximate backend can/can't beat exact). Release config; not a ctest.
add_executable(optics_approx_probe Benchmark/approx_probe.cpp)
target_link_libraries(optics_approx_probe PRIVATE optics::optics)

# Precompute vs OnDemand neighbor-acquisition timing + memory (OnDemand scales past the
# Precompute memory wall). Release config; not a ctest.
add_executable(optics_mode_compare Benchmark/mode_compare.cpp)
target_link_libraries(optics_mode_compare PRIVATE optics::optics)

# Unique-point / weighted OPTICS speedup (#46): full-cloud vs deduplicate + weighted ordering on
# CSV clouds (e.g. color images), at the same epsilon. Release config; not a ctest.
add_executable(optics_dedup_probe Benchmark/dedup_probe.cpp)
target_link_libraries(optics_dedup_probe PRIVATE optics::optics)

# sOPTICS vs exact OPTICS: clustering quality (Rand index) + timing on synthetic
# normalized blobs. Release config; not a ctest.
add_executable(optics_soptics_compare Benchmark/soptics_compare.cpp)
target_link_libraries(optics_soptics_compare PRIVATE optics::optics)

# Emits OPTICS + sOPTICS predicted labels for a CSV cloud (for tools/quality_benchmark.py
# to score ARI/NMI/Rand vs ground truth). Release config; not a ctest.
add_executable(optics_quality_compare Benchmark/quality_compare.cpp)
target_link_libraries(optics_quality_compare PRIVATE optics::optics)

# Emits HDBSCAN* + sHDBSCAN predicted labels for a CSV cloud (for tools/hdbscan_benchmark.py to
# cross-check against scikit-learn's HDBSCAN + ground truth, #52). Release config; not a ctest.
add_executable(hdbscan_compare Benchmark/hdbscan_compare.cpp)
target_link_libraries(hdbscan_compare PRIVATE optics::optics)

# #55 distance-reuse probe: per-phase timing of the reuse fast path vs a recompute-only
# backend on a dense cloud (defines OPTICS_PROFILE internally). Release config; not a ctest.
add_executable(optics_reuse_probe Benchmark/reuse_probe.cpp)
target_link_libraries(optics_reuse_probe PRIVATE optics::optics)

# sOPTICS distance-reuse probe: CEOs out_sq reuse vs recompute, per-phase timing on a
# higher-D cloud (defines OPTICS_PROFILE internally). Release config; not a ctest.
add_executable(optics_soptics_reuse_probe Benchmark/soptics_reuse_probe.cpp)
target_link_libraries(optics_soptics_reuse_probe PRIVATE optics::optics)

# sOPTICS projection probe (#58): Gaussian vs structured (FHT spinner) projection timing + Rand
# vs exact OPTICS across dimension. Release config; not a ctest.
add_executable(optics_soptics_proj_probe Benchmark/soptics_proj_probe.cpp)
target_link_libraries(optics_soptics_proj_probe PRIVATE optics::optics)

# sOPTICS CEOs-parameter sweep (#58): Rand vs exact on a hard 16-D dataset over (D,k,m), to set the
# high-dimension defaults from data. Release config; not a ctest.
add_executable(optics_soptics_param_probe Benchmark/soptics_param_probe.cpp)
target_link_libraries(optics_soptics_param_probe PRIVATE optics::optics)

# HNSW recall-vs-speed probe (#47): HNSW vs exact / eps-approx nanoflann in high-D. Only
# built with the optional HNSW backend enabled. Release config; not a ctest.
if(OPTICS_ENABLE_HNSW)
    add_executable(optics_hnsw_probe Benchmark/hnsw_probe.cpp)
    target_link_libraries(optics_hnsw_probe PRIVATE optics::optics)
endif()
