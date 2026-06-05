# Roadmap to v0.9.0

Status target: a **fast, dependency-free, installable, well-tested** OPTICS release.
Work happens on branch `release-0.9` (off the `modernize-cpp20` modernization). Task IDs
below refer to the session task tracker.

## Exit criteria (definition of done for 0.9)

- Tier-0 performance optimizations landed, each with a **measured** before/after improvement
  and no regressions against the committed baseline.
- A real test framework (not bare `assert`) with per-case CTest integration, green on
  MSVC + GCC + Clang (and the Boost-backend job).
- Performance-regression harness with a committed baseline.
- Edge cases handled and covered by tests.
- Installable via CMake (`find_package(optics)` / FetchContent) with versioning + CHANGELOG.
- Tag `v0.9.0`.

`#23` (faithful ExtractDBSCAN) is **out of scope for 0.9** — deferred.

---

## Foundations — do these first

### Testing framework — decision: **doctest** (#24)

The suites currently live in a hand-rolled `main()` full of `assert()`. Problems: the first
failure aborts the whole run, there is no per-case reporting or CTest granularity, and
`assert` is compiled out under `NDEBUG` (we work around it today with `/UNDEBUG`).

**Adopt [doctest](https://github.com/doctest/doctest):** a single vendored header (matches the
nanoflann/"vendor one header" ethos), the fastest-compiling C++ test framework, with
`TEST_CASE`/`CHECK`/`SUBCASE` and CTest auto-discovery. It is **test-only** and does **not**
affect the library's zero-dependency guarantee. Migration is mechanical (`assert` → `CHECK`).

*Alternatives considered:* Catch2 (heavier compile; v3 is a compiled lib), GoogleTest (needs
build + link, brings a real dependency). doctest is the best fit for a header-only library.

### Performance tracking — decision: **nanobench + committed baseline** (#25)

> "Implement perf tests first to get a baseline to compare to after each essential change."

Vendor [nanobench](https://github.com/martinus/nanobench) (single header, test-only) and add an
`optics_perf` target that measures, with statistical robustness (median + error %):

- **Hot-path microbenchmarks:** `compute_core_dist`, seed-queue operations, and the full
  ordering loop at a modest N — these directly quantify the Tier-0 changes.
- **End-to-end scenarios:** 3D and 16D, float and double, Precompute/OnDemand, 1 vs HW threads,
  at sizes that run quickly and repeatably.

Emit machine-readable results, **commit a baseline** under `perf/` (with the dev-machine
hardware noted), and add a tolerance-based compare step. CI runs it **informationally** (runner
hardware varies, so it is not a gate). This harness must exist **before #12** so each essential
optimization is measured before/after. (`#12` is marked blocked-by `#25`.)

*Note:* `optics_perf` is for repeatable regression tracking; the existing `optics_benchmark`
stays for ad-hoc large-scale (1e6–1e7) exploration (#16).

---

## Tier 0 — Performance arc (measure each against the baseline)

1. **#12 Optimize `compute_core_dist`** *(start here, per request).* Drop the per-call
   `vector<size_t>` copy; fill a reused `thread_local` scratch of squared distances and return
   the sqrt of the MinPts-th smallest. Re-run perf, record the delta.
2. **#13 Fuse neighbor-distance computation** — distances are computed twice today
   (`compute_core_dist` + `update`); compute once and reuse.
3. **#14 Replace the `std::set` seed queue with a lazy-deletion heap** — removes per-insert node
   allocation and improves cache behavior in the sequential ordering loop. Likely the largest
   single win after neighbor queries. Re-verify `chi_*` and the neighbor-mode parity test.
4. **#15 CSR-flatten precomputed neighbors** — one flat index buffer + offsets instead of
   `vector<vector<size_t>>`; cuts millions of allocations and memory at 1e6–1e7 points.
5. **#16 Benchmark at 1e6–1e7 scale** — validate the optimizations and memory at the real
   target sizes; record numbers.

## Tier 1 — Release readiness

6. **#17 CMake install/export + package config** — install headers (incl. vendored nanoflann),
   export `optics::optics`, generate config + version files for `find_package`/FetchContent.
7. **#18 Versioning** — `project(... VERSION 0.9.0)`, `OPTICS_VERSION` macro, `CHANGELOG.md`.
8. **#19 API surface cleanup** — move internal helpers (`bounding_box`, `hypercuboid_volume`,
   `SDA`, `pop_from_set`, …) into `optics::detail`; freeze the public API.
9. **#20 Convenience entry points + byte/int ergonomics** — `extract_dbscan`/`extract_xi`
   wrappers, a one-call cluster→labels helper, and a documented conversion path for `uint8`
   color data (since `T` must be float/double).
10. **#26 License compliance** — nanoflann is **BSD-2-Clause** (its header is preserved in
    `nanoflann.hpp`, satisfying source redistribution); our own license **stays MIT**. Add a
    `THIRD-PARTY-LICENSES` file enumerating nanoflann (BSD-2-Clause) plus the test-only deps
    doctest + nanobench (MIT), and reference it from `README`/`LICENSE`. This also covers the
    BSD binary-distribution attribution clause for downstream consumers.

## Tier 2 — Correctness & confidence (written in doctest)

11. **#21 Brute-force O(n²) reference test** — assert the pipeline matches a naive OPTICS on
    small clouds (the `chi_*` tests only pin current behavior, not paper-correctness).
12. **#22 Edge-case handling + tests** — `< min_pts` points, all-identical points (replace the
    `eps = 1.0` fallback hack), duplicates, empty input, NaN/inf; plus float-vs-double parity
    and seed tie-break determinism.

## Out of scope for 0.9 (parked)

- **#23** faithful ExtractDBSCAN (store `core_dist`, paper-accurate threshold extractor).
- ASan/UBSan CI job, warnings-as-errors, clang-format.
- Doxygen API docs.
- Approximate-NN option for the expensive 16-D regime.

---

## Sequencing

```
Foundations:  #24 doctest        ┐ (can run in parallel)
              #25 perf baseline   ┘  ← must precede #12
Tier 0:       #12 → measure → #13 → #14 → #15 → measure each → #16 (at scale)
Tier 1:       #17 → #18 → #19 → #20
Tier 2:       #21, #22  (in doctest)
Release:      verify checklist → tag v0.9.0
```

## Release checklist

- [ ] doctest suite + Boost-backend test green on MSVC, GCC, Clang.
- [ ] Perf baseline shows net improvement; no regressions beyond tolerance.
- [ ] `find_package(optics)` works from a throwaway consumer project.
- [ ] `THIRD-PARTY-LICENSES` present (nanoflann BSD-2-Clause + test-only MIT deps); README references it.
- [ ] Edge-case tests pass; no `eps = 1.0` hack remaining.
- [ ] `CHANGELOG.md` updated; version bumped; tag `v0.9.0`.
