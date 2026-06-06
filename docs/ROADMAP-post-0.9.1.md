# Lookahead — post-0.9.1 work

A backlog compiled at the end of the 0.9.1 cycle: where to take the library next, grounded in
what the 0.9.1 performance work actually measured (see `perf/README.md`). Nothing here is
committed scope; it is a menu for 0.9.2 / the road to a stable **1.0.0**. Priority tags:
**[P1]** soon / high value, **[P2]** valuable, **[P3]** opportunistic / research.

---

## 1. Getting clever with dense neighborhoods

The headline 0.9.1 finding: on dense clouds (e.g. flat-color images) with auto-`epsilon`, the
average neighborhood grows ~linearly with n, so the ordering is ~**O(n²)** in time and (under
Precompute) memory. Crucially, the cost is in *processing* neighborhoods (`compute_core_dist`
scan + the relax loop, both O(|N_ε|)), **not** the neighbor search — the `optics_approx_probe`
data shows a faster/approximate backend can't help here. So the wins are about **reducing the
work per neighborhood**, not faster NN:

- **[P1] Smarter `epsilon` estimation.** `epsilon_estimation` assumes a *uniform* density over the
  bounding box; on clustered data (every real image) it overestimates ε, which is the root cause of
  the huge neighborhoods. Estimate ε from the **k-distance knee** (sorted k-NN distances, the
  classic DBSCAN heuristic) instead. This alone keeps neighborhoods bounded and sidesteps the O(n²)
  trap for most real inputs — biggest usability+perf lever.
- **[P1] Weighted / unique-point OPTICS.** Color images contain massive point duplication (RGB has
  ≤ 16.7M distinct colors, usually far fewer than the pixel count). Deduplicate to unique points
  carrying a weight (count); make `min_pts`, core-distance, and reachability weight-aware. A 1 MP
  image with ~50k unique colors collapses from 1e6 to 5e4 points — the single biggest win for the
  color use case. Needs careful weighting semantics (a weighted k-th neighbor).
- **[P2] Voxel/grid pre-aggregation** for low-D dense data: bucket points into a grid at ~ε
  resolution and cluster occupied cells as super-points. Approximate but very fast; good for
  color-space previews.
- **[P2] Bounded-neighborhood (approximate) OPTICS:** optionally cap |N_ε| to the k nearest within
  ε, giving O(n·k) regardless of density. Document it as an approximation (changes reachability in
  dense cores).
- **[P2] CSR-flattened Precompute buffer** (old Tier-0 #15, never landed): one flat index array +
  offsets instead of `vector<vector<size_t>>`. Doesn't change the O(n²) asymptotics but cuts
  millions of allocations and the per-vector overhead, and improves cache locality — helps the
  Precompute path that survives for sparse clouds.
- **[P3] Parallelize the relax inner loop.** The OPTICS ordering is inherently sequential, but the
  per-point relax over a huge neighborhood is data-parallel (min-updates into `reachability[]` +
  the seed queue). Tricky to keep the seed order deterministic; research only.

> Note for whoever picks this up: "speed up the neighborhood *search*" is the intuitive framing but
> the wrong target for dense low-D data — the probe shows search is a small slice there. Search
> speed matters in **high** dimensions (§2).

## 2. Backends & other things to test

- **[P1] HNSW** (e.g. hnswlib) as an approximate graph backend for the high-D regime (the 16-D
  perspective-transform target). nanoflann's KD-tree degrades in 16-D; HNSW is state-of-the-art ANN
  and should beat both nanoflann and the current eps-approx backend there. Measure recall vs speed
  with `optics_approx_probe`-style tooling.
- **[P2] Runtime-configurable approximation.** `ApproxNanoflannBackend`'s eps is a *compile-time*
  template parameter today; expose it (and leaf size) as constructor/config values. This needs the
  backend-construction seam generalized — `compute_reachability_dists` currently builds `Backend(points)`
  with no extra args; thread a small backend-config through, or accept a pre-built backend.
- **[P2] Other ANN/spatial backends to benchmark:** FLANN (autotuned), Annoy, ScaNN; a grid/LSH
  backend for low-D dense; cover-trees / VP-trees for high-D *exact*.
- **[P3] GPU backend** (CUDA/SYCL) for the neighbor-query phase on very large clouds.
- **[P2] Approx-backend sweep study:** extend `optics_approx_probe` into a recall/speed Pareto
  across dimensions and eps levels, to pick sane defaults per dimensionality.

## 3. Broadening the test infrastructure

- **[P1] Backend matrix in CI:** run the unit suite against `ApproxNanoflannBackend` and
  `BoostRTreeBackend`, not just the default — today only `boost_backend_tests` exercises Boost.
- **[P1] Perf regression gate:** `optics_perf` is informational; add a tolerance-based compare step
  on the key microbenchmarks (back-to-back runs, not vs an old baseline — see `perf/README.md`).
- **[P2] Property-based / fuzz tests:** random clouds with invariant checks (ordering length == n;
  every reachability is −1 or ≥ 0; threshold/Xi partitions are valid and cover all points).
- **[P2] Sanitizer coverage:** extend ASan/UBSan to the OnDemand path; add **TSan** for the
  Precompute thread pool (`detail/thread_pool.hpp`).
- **[P2] Memory test:** turn `optics_mode_compare` into an assertion that OnDemand peak memory stays
  ~flat while Precompute grows with n (guards the OnDemand invariant that motivated the default flip).
- **[P3] Larger brute-force reference** across more dims/`min_pts`; cross-platform ordering
  determinism pinned in CI (MSVC/GCC/Clang); code-coverage reporting; a golden-CSV visual
  regression on the fixtures.

## 4. Loose ends from the 0.9.1 session

- **[P1] Delete/retire the `perf-dense` branch** — its prototype (profiling + knn core-dist) is
  superseded by the tested `CoreDistMode::Knn` (#24). It is local-only and now stale.
- **[P2] Re-add an optional phase profiler.** The `OPTICS_PROFILE` breakdown (index build /
  precompute / core_dist / relax / loop) from `perf-dense` was genuinely useful and was dropped when
  #24 landed cleanly. Re-introduce it as a documented compile flag (not a bare `#ifdef`).
- **[P2] CMake `CMP0167` warning** when configuring with Boost (the removed `FindBoost` module).
  Set the policy / consume `BoostConfig` cleanly so the Boost build is warning-free.
- **[P3] 400k+ never measured to completion** — Precompute OOMs (~300 GB projected) and OnDemand at
  400k would take ~9 min; only 200k OnDemand was run. Capture a 400k OnDemand point if a scaling
  curve to that size is wanted.
- **[P3] `cluster_csv` has 8 positional args** — migrate to named flags before it grows further.
- **[P3] Example overlap:** `examples/color_clustering` and `examples/cluster_csv` both hand-roll
  CSV I/O; factor a shared reader (the benchmarks already share `test/Benchmark/csv_points.hpp`).
- **[P3] Test the `min_cluster_frac` × Xi interaction** (size-filtering Xi clusters) — added in
  0.9.1, not yet pinned by a case.
- Personal-image timing figure (`docs/img/timing_images_jpg.png`) is intentionally untracked; the
  committed example uses the standard PPM test images.

## 5. Toward a production-ready, stable 1.0.0

- **[P1] Commit to API stability + semver.** Freeze the public surface (`optics::` excluding
  `detail::`), document what is stable, and treat `detail::` as explicitly unstable. 1.0 is a
  promise — the default-mode flip in 0.9.1 is the kind of breaking change that should not happen
  after it.
- **[P1] Pre-allocation guard.** We hit a real ~19 GB Precompute allocation and projected ~300 GB at
  400k. Before committing to a Precompute buffer that would exceed available RAM, warn / throw with
  a clear message (and suggest OnDemand), rather than thrashing or OOM-killing the process.
- **[P1] Smarter ε + weighted points** (from §1) — the two changes that make OPTICS *just work* on
  real images without the O(n²) surprise; arguably gating for a credible 1.0 on the stated targets.
- **[P2] Faithful ExtractDBSCAN.** Store core-distance and offer the paper-accurate threshold
  extractor; today's threshold cut is a documented simplification (parked as #23 in the 0.9
  roadmap). A 1.0 should provide the real thing alongside it.
- **[P2] HDBSCAN-style extraction** (cluster stability / `condensed tree`) as an additional method —
  popular, and complements the ξ extractor.
- **[P2] Progress + cancellation callback** for long runs (large clouds take minutes); essential for
  interactive/production use.
- **[P2] Packaging:** vcpkg and Conan ports, a single-header amalgamation for drop-in use, and the
  hosted Doxygen site (the `docs` target exists). The `find_package`/FetchContent path is already in.
- **[P3] Input validation & clear errors** across the board (NaN/inf coordinates, empty/degenerate
  inputs have partial coverage); documented tie-break/determinism guarantees.
- **[P3] Research: parallel OPTICS** (POPTICS-style) — the sequential ordering is the scaling
  ceiling once neighborhoods are bounded; the long-term path to multi-core scaling.
