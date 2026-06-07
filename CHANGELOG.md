# Changelog

All notable changes to OPTICS-Clustering are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Renamed `cluster_dbscan` → `cluster_threshold`** (C++ and the Python binding): the flat cut is
  the OPTICS paper's *ExtractDBSCAN* (the same clustering DBSCAN gives at `eps = threshold`), not a
  DBSCAN run. `cluster_dbscan` stays as a deprecated alias.
- `cluster_threshold` and `extract_xi` are now a parallel pair, both `(points, min_pts, [param])`:
  `extract_xi` takes the **point cloud** (was a cluster-ordering) and computes the ordering itself,
  and `threshold`/`chi` are **optional** — an omitted `threshold` uses an educated default (a high
  percentile of the reachabilities); `chi` defaults to `0.05`.
- Runtime-millisecond outputs (examples, benchmarks, timing scripts) now round **up** to the next
  whole millisecond.

## [0.9.1] — 2026-06-06

Focus: make the v0.9.0 core **adoptable** — a fast first run on your own data, an honest
comparison to other algorithms, independent validation, and backend/perf hardening. See
`docs/ROADMAP-0.9.1.md` (GitHub milestone V0.9.1).

### Added
- `CoreDistMode::Knn` — a k-NN core-distance path (new `core_dist_mode` argument of
  `compute_reachability_dists`, default `Scan`) that avoids scanning huge eps-neighborhoods
  on dense clouds (~27% faster end-to-end on a flat-color-like cloud) with identical
  results. Backends opt in via the `KnnCoreDist` concept (#24).
- `ApproxNanoflannBackend` — eps-approximate nearest-neighbor backend for the
  high-dimensional regime; `NanoflannBackend` gained a compile-time approximation knob
  (0 = exact, the default) (#28).
- `examples/cluster_csv` — a generic "cluster your own CSV" example (2/3/4/16-D) writing
  labeled-points + reachability CSVs for `tools/visualize.py` (#25).
- Python tooling: `tools/datasets.py` (reproducible 2-D datasets, incl. a varying-density
  case), `compare_algorithms.py` (OPTICS vs k-means vs DBSCAN figure, with the
  different-density case OPTICS wins via Xi), `validate_sklearn.py` (cross-check against
  scikit-learn OPTICS), `timing_compare.py` + the `optics_backend_compare` harness
  (this library is ~100–800× faster than scikit-learn OPTICS across the internal
  backends), `requirements.txt`, and one-command `scripts/demo.{ps1,sh}`.
- Timing harnesses default to **4 worker threads** (override with `OPTICS_BENCH_THREADS`)
  for reproducible numbers; `cluster_csv` gained `xi_chi` (hierarchical extraction) and
  `n_threads` arguments.
- `optics_py` — optional pybind11 binding for 1/2/3/4-D NumPy clouds
  (`OPTICS_BUILD_PYTHON`, off by default) (#23).
- README: an honest OPTICS-vs-k-means-vs-DBSCAN comparison, a "run on your own data"
  quickstart, and guides for reading the reachability plot and choosing parameters, with
  committed figures (#29, #30).
- `CITATION.cff`, a Doxygen `docs` target, and an ASan/UBSan CI job; warnings-as-errors on
  the MSVC CI job (`OPTICS_WARNINGS_AS_ERRORS`) (#34).
- Perf harness: dense-neighborhood (Scan vs Knn) and backend-comparison scenarios; baseline
  refreshed (#26).
- Real-image color-clustering runtime analysis (`tools/timing_images.py`) and three new
  benchmark harnesses: `optics_approx_probe` (approximate-backend recall vs eps/dimension),
  `optics_mode_compare` (Precompute vs OnDemand time + memory), and the scaling-by-sample-size
  study — documented in `perf/README.md`, including *why the approximate backend helps only in
  high dimensions* and the dense-neighborhood O(n²) memory wall.

### Changed
- **`compute_reachability_dists` and `cluster_dbscan` now default to `NeighborMode::OnDemand`**
  (was `Precompute`). OnDemand uses O(one neighborhood) memory instead of O(n × avg_neighbors), so
  it never OOMs on dense data and is ~30% faster there; Precompute (faster on sparse/low-density
  clouds) is now opt-in. Results are identical; only the time/memory profile changes. `cluster_csv`
  selects Precompute when `n_threads > 1`.

### Fixed
- Backend equivalence is now asserted end-to-end (nanoflann ↔ Boost produce identical
  orderings), not just matching neighbor sets (#27).
- A discarded nanoflann `radiusSearch` return and a shadowed test variable (clean under
  warnings-as-errors); vendored nanoflann is included with its warnings suppressed.

## [0.9.0] — 2026-06-05

First modernized release: a fast, dependency-free, cross-platform C++20 library.

### Added
- C++20 runtime API `compute_reachability_dists<T, Dim, Backend>(points, min_pts,
  epsilon = -1, mode = Precompute, n_threads = 0)` — coordinate type and dimension
  are deduced from the cloud; no compile-time point count.
- Swappable neighbor-search backend via a `NeighborSearch` concept. Default
  `NanoflannBackend` (zero-copy adaptor over the user's `vector<array<T,Dim>>`);
  optional `BoostRTreeBackend` behind `OPTICS_ENABLE_BOOST_RTREE`.
- `NeighborMode::Precompute` (parallel) / `OnDemand` (lean memory) acquisition with a
  portable thread pool (`n_threads`, 0 ⇒ hardware concurrency).
- Dimension-agnostic CSV export (`optics/io.hpp`) and `tools/visualize.py`
  (matplotlib 2D/3D/PCA scatter + reachability plot).
- N-dimensional synthetic-data generators (`optics/testdata.hpp`).
- Convenience helpers: `cluster_dbscan`, `extract_xi`, and `convert_cloud<float>`
  (uint8/int → float, e.g. for color data).
- `examples/color_clustering` — end-to-end 3-D color-space clustering of an image
  (mean-color points + enclosing spheres; interactive / PNG / plotly output) plus a
  k-means timing comparison.
- `optics/version.hpp` (`OPTICS_VERSION*`, `optics::version()`).
- CMake presets (msvc / linux-gcc / linux-clang) and a GitHub Actions CI matrix
  (Linux GCC/Clang, Windows MSVC, plus a Boost-backend job).
- Test/benchmark tooling: doctest unit suite, nanobench perf-regression harness with a
  committed baseline, and a 1e6–1e7 scale harness.

### Changed
- Removed all external dependencies (FunctionalPlus, CrikeeIP/Geometry). Only nanoflann
  (BSD 2-Clause) is vendored; everything else is the standard library.
- Default neighbor search is nanoflann (refreshed to the latest upstream).
- Library errors throw exceptions instead of calling `std::exit`.

### Removed
- The custom compile-time-sized `kdt::KDTree` backend and the `n_points` template
  parameter it forced.
- Built-in C++ image rendering (replaced by CSV export + matplotlib).

### Performance
- `compute_core_dist`: reuse a thread-local distance buffer — ~2.2–2.7× on the hot path.
- Seed queue: lazy-deletion min-heap instead of `std::set` — ~14–17% faster ordering.
- Validated to 1e7 points (3D); see `perf/README.md`.

### Fixed
- nanoflann's L2 metric works in squared distance; the backend now squares the search
  radius (the old nanoflann path searched an unsquared radius).
- `epsilon_estimation` uses only the non-degenerate dimensions, so collinear/planar inputs
  get a sensible auto-epsilon instead of collapsing to a magic `1.0` fallback (which now
  applies only to truly all-identical points, where the value is immaterial).
