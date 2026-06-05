# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A header-only C++20 implementation of the **OPTICS** density-based clustering algorithm (Ankerst et al., 1999). Given points in R^n it produces the cluster-ordering + reachability distances, from which clusters are extracted — without knowing the number of clusters a priori. Built for time-critical use on large clouds (e.g. 3-D color spaces at 1e6–1e7 points, 16-D perspective transforms at 1e5–1e6).

The library has **no mandatory external dependencies**: only the vendored `nanoflann` (in `include/optics/`) plus the standard library. Boost is an *optional* alternative backend, off by default.

## Build & test

Requires a C++20 compiler (MSVC 2022, GCC 10+, or Clang 13+) and CMake ≥ 3.21. Use the presets:

```sh
cmake --preset msvc        # or: linux-gcc, linux-clang
cmake --build --preset msvc
ctest --preset msvc
```

Without presets (e.g. a specific config):
```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

- **Run one suite:** `ctest --preset msvc -R optics_tests` (suites: `optics_tests`, `optics_visual_test`), or run the built exe directly (`build/test/Release/optics_tests.exe`). Use `ctest --output-on-failure` to see doctest's per-case report.
- **Unit suite uses [doctest](https://github.com/doctest/doctest)** (`test/third_party/doctest.h`, vendored, test-only): `optics_tests` is `TEST_CASE`/`CHECK`-based, so assertions run regardless of `NDEBUG`. Note: doctest can't decompose `&&`/`||` in a `CHECK` — wrap such expressions in parens. The separate `optics_visual_test` still verifies via `assert()`, so it keeps `/UNDEBUG` (MSVC) / `-UNDEBUG` (GCC/Clang).
- **Optional Boost backend:** configure with `-DOPTICS_ENABLE_BOOST_RTREE=ON` (needs Boost.Geometry). This compiles `BoostRTreeBackend` and an `#ifdef`-gated equivalence test. Boost is exercised in CI (`.github/workflows/ci.yml`), not in the default build.
- **Benchmark:** build the `optics_benchmark` target (Release) and run it; an optional integer arg scales the point counts (`optics_benchmark 4`). Not a ctest.
- **Local toolchain note:** on the original dev machine only VS2022 is installed (no g++/standalone cmake on PATH); use its bundled cmake at `…\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

## Architecture & data flow

The pipeline lives under `include/optics/`. Public entry point is `optics.hpp`.

1. **`compute_reachability_dists<T, Dim, Backend = NanoflannBackend<T,Dim>>(points, min_pts, epsilon = -1, mode = Precompute, n_threads = 0)`** → `std::vector<reachability_dist>` (the cluster-ordering; `reach_dist == -1` means UNDEFINED/unreached). `T` is `float`/`double`, points are `std::vector<std::array<T,Dim>>`. `epsilon <= 0` auto-estimates via `epsilon_estimation`.
2. **Extraction:**
   - Threshold cut: `get_cluster_indices(reach_dists, threshold)` → flat clusters (noise → singletons; a deliberate simplification of the paper's ExtractDBSCAN, which also used core-distance).
   - Xi / steep-area: `get_chi_clusters_flat(...)` → flat `(begin,end)` ranges; `get_chi_clusters(...)` builds the nested `cluster_tree` (`tree.hpp`). This is paper Defs 9–11 / Fig. 19 and the most subtle code; its behavior is pinned by the `chi_test_*` cases.
3. **Inspection (optional, no rendering in C++):** `io.hpp` exports dimension-agnostic CSV (`export_points_csv` with `cluster_labels`, `export_reachability_csv`); `tools/visualize.py` renders 2D/3D/PCA scatter + reachability plots with matplotlib.

### Neighbor-search backend seam
Backends satisfy the `NeighborSearch` C++20 concept (`backend.hpp`): they ingest the cloud once at construction and answer `radius_search(point, r, out)` with **no per-query point conversion**.
- `NanoflannBackend` (default): a zero-copy adaptor over the user's `std::vector<std::array<T,Dim>>`. **nanoflann's L2 metric works in squared distance** — the backend squares the radius internally.
- `BoostRTreeBackend` (`boost_backend.hpp`, behind `OPTICS_ENABLE_BOOST_RTREE`): box pre-filter + exact Euclidean check.
Swapping backends is a single template argument; call sites don't reshape data.

### Neighbor acquisition & parallelism
`NeighborMode::Precompute` (default) queries every point's neighbors up front in parallel via `detail/thread_pool.hpp`'s `parallel_for` (`n_threads = 0` ⇒ hardware concurrency); `OnDemand` queries lazily for lean memory on huge clouds. The OPTICS ordering loop itself is sequential; the ε-query phase is the hotspot, so parallel precompute is the main speed lever (the benchmark shows ~10× in 16-D).

Supporting headers: `detail/math.hpp` (distance/`pi`/`in_range`, replacing the old Geometry dep), `testdata.hpp` (N-dim synthetic clouds, shared with the benchmark), `test/support/ppm_fixture.hpp` (P6 reader for hand-drawn 2-D fixtures), `Stopwatch.hpp` (benchmark timing).

## Gotchas

- **`epsilon` is a `double` parameter, deliberately not `T`** — making it `T` lets it participate in template deduction and clashes with `T` deduced from `points` (e.g. passing the literal `10`).
- `T` must be `float`/`double` (`static_assert`); integer coordinate inputs must be converted first.
- The library throws (`std::invalid_argument`/`std::runtime_error`) instead of calling `std::exit`.
- 16-D nearest-neighbor is genuinely expensive (curse of dimensionality); prefer `Precompute` + threads, and use the benchmark to pick a backend.
- `compute_core_dist` copies the neighbor index list per call — fine for realistic neighborhoods, a known target for future optimization on very dense data.
