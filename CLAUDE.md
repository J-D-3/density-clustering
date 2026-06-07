# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A header-only C++20 implementation of the **OPTICS** density-based clustering algorithm (Ankerst et al., 1999). Given points in R^n it produces the cluster-ordering + reachability distances, from which clusters are extracted — without knowing the number of clusters a priori. Built for time-critical use on large clouds (e.g. 3-D color spaces at 1e6–1e7 points, 16-D perspective transforms at 1e5–1e6).

The library has **no mandatory external dependencies**: only the vendored `nanoflann` (in `include/optics/`) plus the standard library. Boost is an *optional* alternative backend, off by default.

For project history and direction: `CHANGELOG.md` (release notes, current version 0.9.1) and `docs/ROADMAP-*.md` (planned work — e.g. the `chi_tree_to_points` helper is queued in `docs/ROADMAP-post-0.9.1.md`). `THIRD-PARTY-LICENSES.md` covers vendored-dep licensing; `CITATION.cff` is the citation metadata.

For algorithm research and future direction: `documentation/` holds papers + `references.md` for the planned **sOPTICS** (sDBSCAN/sOPTICS, random-projection OPTICS — GitHub issue #50). The 1.0.0 milestone tracks the bigger algorithm/benchmark work: sOPTICS (#50), HDBSCAN\* (#52), CPU-speed comparisons vs ELKI/dbscan/sDbscan (#53), an ARI/NMI/Rand quality harness (#54), and non-Euclidean metrics (#51). Original OPTICS/FOPTICS papers live in `background/`.

Before tagging 1.0.0, one large reproducible study (the **benchmark matrix**) picks the library's data-dependent defaults (backend by dimensionality, sOPTICS-vs-OPTICS, neighbor mode/backend, epsilon estimator) and becomes the citable performance reference. Its design — axes, the tiered/fractional reduction, which existing harnesses feed it, comparability rules — is in `docs/ROADMAP-1.0.0-benchmark-matrix.md`.

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
- **Benchmarks (Release, none are ctests):** `optics_benchmark` — general benchmark; optional integer arg scales the point counts (`optics_benchmark 4`). `optics_perf` — nanobench perf-regression harness, emits `optics_perf.csv`. `optics_scale` — large-scale (1e6–1e7) 3-D timing harness.
- **Phase profiler:** build any consumer with `-DOPTICS_PROFILE` (MSVC `/DOPTICS_PROFILE`) to print a per-phase timing breakdown (index build / precompute / core_dist / relax / loop) to stderr after each `compute_reachability_dists` call. Off by default ⇒ zero overhead and no call-site `#ifdef` (the conditional compilation lives in `include/optics/detail/profile.hpp`).
- **Examples:** built when `OPTICS_BUILD_EXAMPLES` is ON (default for a top-level build). `optics_color` (`examples/color_clustering/`) clusters a 3-D RGB color space; `cluster_csv` (`examples/cluster_csv/`) is a generic CSV cloud clusterer (2/3/4/16-D). Each dir has its own `README.md` and companion Python scripts. Toggle build scope with `OPTICS_BUILD_TESTS` / `OPTICS_BUILD_EXAMPLES`; `OPTICS_BUILD_PYTHON` (OFF) builds the optional pybind11 NumPy binding under `python/` (see `python/README.md`); `OPTICS_INSTALL` adds install + `find_package(optics)` / FetchContent config.
- **Local toolchain note:** on the original dev machine only VS2022 is installed (no g++/standalone cmake on PATH); use its bundled cmake at `…\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

## Architecture & data flow

The pipeline lives under `include/optics/`. Public entry point is `optics.hpp`.

1. **`compute_reachability_dists<T, Dim, Backend = NanoflannBackend<T,Dim>>(points, min_pts, epsilon = -1, mode = OnDemand, n_threads = 0, core_dist = Scan)`** → `std::vector<reachability_dist>` (the cluster-ordering; `reach_dist == -1` means UNDEFINED/unreached). `T` is `float`/`double`, points are `std::vector<std::array<T,Dim>>`. `epsilon <= 0` auto-estimates via the **k-distance-knee** estimator (`epsilon_estimation_knee`, the default since it tracks the within-cluster scale; the uniform-density `epsilon_estimation` over-estimates on clustered data — issue #57). It falls back to the uniform estimate for backends without `KnnCoreDist`, or on degenerate (zero-variance) inputs. `mode` defaults to **OnDemand** (see below); `core_dist = Knn` (a `CoreDistMode`) is an equivalent, faster-on-dense core-distance path requiring a backend that models `KnnCoreDist`.
2. **Extraction:**
   - Threshold cut: `get_cluster_indices(reach_dists, threshold)` → flat clusters (noise → singletons; a deliberate simplification of the paper's ExtractDBSCAN, which also used core-distance).
   - Xi / steep-area: `get_chi_clusters_flat(...)` → flat `(begin,end)` ranges; `get_chi_clusters(...)` builds the nested `cluster_tree` (`tree.hpp`). This is paper Defs 9–11 / Fig. 19 and the most subtle code; its behavior is pinned by the `chi_test_*` cases.
   - One-call convenience wrappers, both `(points, min_pts, [param])`: `cluster_threshold(points, min_pts, threshold = auto)` (compute + flat cut; the paper's ExtractDBSCAN — `threshold < 0` ⇒ `detail::default_threshold`, a high reachability percentile; `cluster_dbscan` is a `[[deprecated]]` alias), `extract_xi(points, min_pts, chi = 0.05)` (compute + Xi, flattened — use `get_chi_clusters` for the tree), and `convert_cloud<Out>(in)` to lift an integer/byte cloud (e.g. `uint8` color data) to `float`/`double` before clustering.
3. **Inspection (optional, no rendering in C++):** `io.hpp` exports dimension-agnostic CSV (`export_points_csv` with `cluster_labels`, `export_reachability_csv`); `tools/visualize.py` renders 2D/3D/PCA scatter + reachability plots with matplotlib. The `tools/` dir has its own `README.md` and further Python helpers (synthetic `datasets.py`, scikit-learn cross-checks `validate_sklearn.py`, algorithm-comparison figures `compare_algorithms.py`).

### Neighbor-search backend seam
Backends satisfy the `NeighborSearch` C++20 concept (`backend.hpp`): they ingest the cloud once at construction and answer `radius_search(point, r, out)` with **no per-query point conversion**.
- `NanoflannBackend<T,Dim,ApproxEpsPermille=0>` (default): a zero-copy adaptor over the user's `std::vector<std::array<T,Dim>>`. **nanoflann's L2 metric works in squared distance** — the backend squares the radius internally. Also exposes `knn_core_dist` (for `CoreDistMode::Knn`). A non-zero `ApproxEpsPermille` enables nanoflann's eps-approximate search; `ApproxNanoflannBackend<T,Dim,Permille=100>` is the alias.
- `BoostRTreeBackend` (`boost_backend.hpp`, behind `OPTICS_ENABLE_BOOST_RTREE`): box pre-filter + exact Euclidean check.
Swapping backends is a single template argument; call sites don't reshape data. The three internal backends are pinned interchangeable by `boost_backend_tests` (Nanoflann/Boost bit-identical ordering) and the approx recall test.

### Neighbor acquisition (mode = OnDemand by default, Precompute opt-in)
`NeighborMode::OnDemand` (**the default since v0.9.1**) queries one neighborhood at a time during the ordering — **O(one neighborhood) memory**. `Precompute` queries every point's neighbors up front in parallel via `detail/thread_pool.hpp`'s `parallel_for` (`n_threads = 0` ⇒ hardware concurrency) and caches them — **O(n × avg_neighbors) memory**. Both produce identical orderings (pinned by `neighbor_mode_tests`). The OPTICS ordering loop itself is always sequential. Which mode is faster depends on neighborhood density: **Precompute wins on sparse/low-density clouds** (small cache, parallel queries — e.g. ~10× in 16-D); **OnDemand wins on dense clouds** (e.g. color images — it avoids materializing/re-reading a multi-GB cache, ~30% faster at 100k px, and never OOMs). See `perf/README.md`.

### sOPTICS: approximate ordering via random projections (the provider seam)
`compute_soptics_reachability_dists(points, min_pts, epsilon = -1, n_projections = 1024, k = 0, m = 0, seed = 42, n_threads = 0)` is a scalable, **approximate** OPTICS (sDBSCAN/sOPTICS, Xu & Pham 2024; issue #50, reimplemented from the paper in `documentation/`). It deliberately does **not** use the `NeighborSearch` backend seam: instead of a radius search it builds **CEOs random-projection neighborhoods** (`detail/random_projection.hpp` — project onto `n_projections` Gaussian vectors, take each point's top-`k` extreme vectors, gather the top-`m` extreme points of each, ε-filter, symmetrize) and feeds them — with core-distance = the `min_pts`-th nearest candidate — into the **shared ordering driver `detail::optics_order`**. That driver is the algorithm-agnostic loop (seed queue, relaxation, lazy deletion) factored out of `compute_reachability_dists`; both OPTICS and sOPTICS drive it through a neighbor-provider + core-dist-provider pair (template params, so the `KnnCoreDist` `if constexpr` stays zero-overhead and orderings remain byte-identical for OPTICS). The metric is **cosine** (points are L2-normalized onto the unit sphere, where Euclidean distance is monotone in cosine distance), so it is not raw-Euclidean OPTICS (general metrics: issue #51). Output is randomized but **deterministic in `seed`** ⇒ validated by Rand/NMI agreement with exact OPTICS, not bit-identical orderings.

Supporting headers: `detail/math.hpp` (distance/`pi`/`in_range`, replacing the old Geometry dep), `testdata.hpp` (N-dim synthetic clouds, shared with the benchmark), `test/support/ppm_fixture.hpp` (P6 reader for hand-drawn 2-D fixtures), `Stopwatch.hpp` (benchmark timing).

## Gotchas

- **`epsilon` is a `double` parameter, deliberately not `T`** — making it `T` lets it participate in template deduction and clashes with `T` deduced from `points` (e.g. passing the literal `10`).
- `T` must be `float`/`double` (`static_assert`); integer coordinate inputs must be converted first.
- The library throws (`std::invalid_argument`/`std::runtime_error`) instead of calling `std::exit`.
- 16-D nearest-neighbor is genuinely expensive (curse of dimensionality), and there the *search* dominates — opt into `Precompute` + threads, and the `ApproxNanoflannBackend` actually helps (eps-pruning cuts tree traversal: ~2.4× at eps=1.0 with recall still ~1.0). In **low** dimensions the approximate backend does **not** help (recall stays ~1.0, nothing to prune) — see `perf/README.md`.
- **Dense neighborhoods (e.g. flat-color images) are the perf trap.** With auto-`epsilon`, the average neighborhood grows ~linearly with n, so the ordering is ~**O(n²)** in time *and* (under Precompute) memory — at ~100k px the Precompute buffer is ~19 GB. The cost is in *processing* neighborhoods (`compute_core_dist` scan + relax, both O(|nbrs|)), **not** the search, so a faster/approximate backend can't fix it; the auto-eps now defaults to the knee estimator (smaller than the old uniform one, which helps), but on genuinely dense clouds also use `OnDemand` mode (the default), `CoreDistMode::Knn`, an even smaller explicit `epsilon`, or downsample. (`compute_core_dist` no longer copies the index list — it fills a reused `thread_local` of squared distances.)
- Perf tooling lives under `test/Benchmark/`: `optics_perf` (nanobench regression), `optics_benchmark`/`optics_scale` (timing), `optics_backend_compare` (per-backend, paired with `tools/timing_compare.py`/`timing_images.py` for scikit-learn), `optics_approx_probe` (approx recall vs eps/dim), `optics_mode_compare` (Precompute vs OnDemand). All default to 4 threads via `bench::threads()` (override `OPTICS_BENCH_THREADS`).
