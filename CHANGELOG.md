# Changelog

All notable changes to OPTICS-Clustering are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.9.0] — unreleased

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
