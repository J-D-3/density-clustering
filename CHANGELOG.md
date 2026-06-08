# Changelog

All notable changes to OPTICS-Clustering are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Optional HNSW approximate backend** `HnswBackend<T,Dim>` (`include/optics/hnsw_backend.hpp`, behind
  `OPTICS_ENABLE_HNSW`, OFF by default; vendored header-only hnswlib under `include/optics/hnswlib/`,
  Apache-2.0). Models the `NeighborSearch` concept (radius search via grow-k k-NN + filter) and the optional
  `KnnCoreDist` capability. **Honest finding from the new `optics_hnsw_probe` recall/speed harness:** at the
  library's target dimensionalities (≤32-D, up to ~50k points) the exact nanoflann KD-tree is *faster* than
  HNSW for radius search despite HNSW's ~1.0 recall — the KD-tree doesn't degrade until much higher D, and the
  radius access pattern doesn't suit HNSW's k-NN-native design. So HNSW is an opt-in very-high-D escape hatch,
  not a 16-D default. Gated test verifies the concept + recall + end-to-end clustering (#47).
- **Non-Euclidean metrics for sOPTICS** (`Metric` enum + `metric` / `kernel_scale` arguments on
  `compute_soptics_reachability_dists`): `Metric::L2` and `Metric::L1` embed points into random Fourier
  features (`include/optics/detail/random_features.hpp`) whose cosine similarity approximates the
  Gaussian / Laplacian kernel (Rahimi & Recht 2007), then run the existing cosine CEOs/sOPTICS pipeline
  on the features — so the cluster-ordering tracks Euclidean / Manhattan distance on the *original* data.
  `kernel_scale <= 0` uses a median-distance auto bandwidth. `Metric::Cosine` (default) is unchanged.
  χ² / Jensen-Shannon are not yet implemented (they need non-negative histogram inputs and a different
  feature construction) — see #51. Validated by Rand-index agreement with exact Euclidean OPTICS and
  seed-determinism (#51).
- **Distance reuse on the exact OPTICS path** (#55): a new optional `RadiusSearchWithDists` backend capability
  lets the ordering reuse the squared distances the neighbor search already computed, instead of recomputing
  them in the core-distance scan *and* the relaxation. `NanoflannBackend` models it for `double` coordinates,
  where its squared distance is bit-identical to `detail::square_dist`, so OPTICS orderings are byte-for-byte
  unchanged; `float` and capability-less backends keep the recompute path. Detected with `if constexpr` — no
  API change, automatic. On a dense 3-D cloud (n=24k, `optics_reuse_probe`) this cut the core-distance phase
  ~30% and the relaxation ~23% (~12% off the ordering loop), with identical orderings. The `detail::optics_order`
  driver gains a third (distance) provider; the recompute default is byte-identical to the previous inline call.
- **Distance reuse for sOPTICS** (the same idea on the random-projection path): `ceos_neighbors` optionally
  returns each candidate's squared distance — the one its ε-filter already computed — parallel to the neighbor
  lists (`out_sq`), and `compute_soptics_reachability_dists` reuses them in the core-distance scan and the
  relaxation instead of recomputing. Because every distance comes from `detail::square_dist` (accumulated in
  `double` regardless of `T`), the ordering is **byte-identical for both float and double** sOPTICS — no
  double-only gate is needed (unlike the backend path). On a 16-D cloud (n=18k, `optics_soptics_reuse_probe`)
  this cut the core-distance phase ~36% and the relaxation ~39% (~34% off the loop), orderings identical; the
  win grows with dimension (e.g. the L2/L1 metric embeddings of #51).
- `optics::compute_soptics_reachability_dists` — **sOPTICS**, a scalable, approximate OPTICS via CEOs
  random projections (sDBSCAN/sOPTICS, Xu & Pham, NeurIPS 2024), reimplemented from the paper. Cosine
  metric (points are L2-normalized onto the unit sphere internally); returns the same
  `reachability_dist` cluster-ordering, so all existing extraction (threshold / Xi) applies unchanged.
  Output is randomized but deterministic in `seed`, and validated by Rand-index agreement with exact
  OPTICS (not bit-identical orderings). New `include/optics/detail/random_projection.hpp` holds the
  CEOs neighbor index (#50).
- Internal: the OPTICS ordering loop is factored into `detail::optics_order`, an algorithm-agnostic
  driver (seed queue / relaxation / lazy deletion) shared by OPTICS and sOPTICS through a
  neighbor-provider + core-distance-provider pair. Behavior-preserving for OPTICS (orderings unchanged).
- `documentation/` — archived sources for the random-projection work (the sDBSCAN/sOPTICS paper plus a
  `references.md` of citations, comparison targets, and licensing notes).
- Benchmarks: `optics_soptics_compare` (sOPTICS vs exact OPTICS — Rand index + timing on synthetic
  normalized blobs) and `optics_quality_compare` (emits OPTICS/sOPTICS predicted labels for a CSV cloud).
- `tools/quality_benchmark.py` — a clustering-quality harness scoring OPTICS + sOPTICS (ours),
  scikit-learn OPTICS + HDBSCAN, and **mhahsler/dbscan (R)** against ground-truth labels with
  **ARI / NMI / Rand**, plus a timing table (#54, #53). dbscan-R runs as an exact-Euclidean OPTICS
  at the *same* generating distance as ours (fair timing) when R + the `dbscan` package are present
  (auto-detected; gracefully skipped otherwise). Remaining external engines (ELKI, sDbscan) are
  documented in `tools/README.md`.
- `tools/fetch_datasets.py` + `tools/run_dbscan_r.R` — fetch Franti's third-party benchmark "shape
  sets" (Aggregation/Compound/spiral/R15/jain/flame/D31) and run mhahsler/dbscan's OPTICS+Xi; both
  used by the quality harness.
- Optional `min_cluster_size` parameter on `get_chi_clusters_flat` / `get_chi_clusters` / `extract_xi`
  — decouples the Xi extractor's minimum-cluster-size / steep-area span cap from `min_pts` (ELKI
  parity). `0` (default) uses `min_pts`, so existing behavior and the `chi_test_*` cases are unchanged
  (#57).
- **Benchmark finding (#57):** the under-segmentation of clustered data (starkly R15: ARI 0.43) is the
  uniform-density `epsilon_estimation` over-shooting, not the Xi logic; `epsilon_estimation_knee` (#41)
  recovers it (R15 ARI 0.95). `tools/quality_benchmark.py` now defaults to `--eps knee`. See
  `docs/benchmarking.md`.

### Changed
- **Auto-epsilon now defaults to the k-distance-knee estimator** (`epsilon_estimation_knee`) instead of
  the uniform-density `epsilon_estimation`, across `compute_reachability_dists` and the `cluster_threshold`
  / `extract_xi` wrappers. The uniform estimate over-shoots on clustered data, which slows the dense path
  and over-smooths the reachability so Xi under-segments (Franti R15: ARI 0.43 → 0.95). It falls back to
  the uniform estimate for backends without `KnnCoreDist` and on degenerate (zero-variance) inputs. Pass
  an explicit `epsilon` for the old behavior. A deliberate pre-1.0 behavior change toward the API freeze
  (#49, #57).

## [0.9.2] — 2026-06-07

Focus: usability and hardening on the road to 1.0 — smarter parameters, safer memory, and broader
test/CI coverage. See `docs/ROADMAP-post-0.9.1.md` (GitHub milestone 0.9.2).

### Added
- `optics::epsilon_estimation_knee` — an opt-in k-distance-knee (classic DBSCAN) epsilon estimator
  that avoids the uniform-density over-estimate of `epsilon_estimation` on clustered data. Reuses the
  backend k-NN path; the default auto-epsilon is unchanged (#41).
- `optics::chi_tree_to_points` — maps a Xi `cluster_tree` onto the same shape with each node holding
  its list of original point indices, **preserving nesting**, so a hierarchy is as easy to consume as
  the flattened `extract_xi` (#36).
- Precompute pre-allocation guard: an optional `max_precompute_bytes` argument to
  `compute_reachability_dists` estimates the neighbor cache from a small sample and throws (suggesting
  `OnDemand`) before an oversized allocation. Default `0` = no check; `OnDemand` is unaffected (#37).
- `OPTICS_PROFILE` compile flag — prints a per-phase timing breakdown (index build / precompute /
  core_dist / relax / loop) to stderr after each `compute_reachability_dists` call. Off by default ⇒
  zero overhead and no call-site `#ifdef` (`include/optics/detail/profile.hpp`) (#42).
- Tests: property/fuzz invariants over random clouds (written generically over the `reachability_dist`
  contract so approximate variants can reuse them) (#44); a cross-backend consistency test
  (nanoflann / approximate / Boost) (#43); a deterministic memory-footprint invariant — Precompute
  cache grows with n while OnDemand stays bounded (#45); and a `min_cluster_frac` × Xi size-filter
  case (#39).
- `examples/shared/csv_io.hpp` — a single CSV reader shared by both example programs, replacing the
  hand-rolled parsing each carried (#38).

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

### Fixed
- Configuring with `-DOPTICS_ENABLE_BOOST_RTREE=ON` no longer emits the CMake `CMP0167` deprecation
  warning: the policy is set to `NEW` (consume `BoostConfig`) and the header-only `Boost::headers`
  target is linked instead of the FindBoost-only `Boost::boost` (#35).

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
