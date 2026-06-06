# Roadmap to v0.9.1

Status target: take the working, dependency-free **v0.9.0** core and make it **adoptable** — a
first-time user can inspect clustering on their own data in minutes, understand *when* to pick
OPTICS over k-means/DBSCAN, and trust it via independent validation — plus backend and
performance hardening.

Work happens on short-lived branches off `master`; milestone is **V0.9.1**
(GitHub milestone #2). Issue IDs below link the tracked work.

## Exit criteria (definition of done for 0.9.1)

- A newcomer can go from clone to a plot of **their own** data via one documented path.
- README states *honestly* where k-means and DBSCAN win, backed by real numbers.
- Backend equivalence test green (Nanoflann ↔ Boost); approximate-NN backend lands with a
  measured speedup and a bounded recall/label-drift test.
- Independent cross-validation against scikit-learn OPTICS passes within tolerance.
- Perf harness gains dense-neighborhood + backend-comparison scenarios; baseline refreshed.
- Suite green on MSVC + GCC + Clang (and the Boost-backend job).
- Tag `v0.9.1`.

The full pybind11 binding (**#23**) is a **stretch** — it may slip to 0.9.2 if it threatens the
milestone.

---

## Tier 0 — Performance & backends

1. **#24 Productionize the perf-dense k-NN core-distance path.** Turn the `knn_core_dist` +
   `OPTICS_PROFILE`/`OPTICS_KNN_CORE` prototype (branch `perf-dense`) into a tested, documented
   option; add a doctest that it matches the eps-scan core distance; measure on the
   dense-neighborhood scenario; merge to `master`. *Start here — already begun.*
2. **#26 Extend the perf harness** — add a dense-neighborhood (flat-color) scenario and a
   Nanoflann-vs-Boost backend-comparison run to `test/Benchmark/perf.cpp`/`optics_benchmark`;
   refresh `perf/baseline.csv` + `perf/README.md`.
3. **#27 Backend equivalence + perf cross-check** — a doctest (gated by
   `OPTICS_ENABLE_BOOST_RTREE`) asserting `NanoflannBackend` and `BoostRTreeBackend` return
   identical neighbor sets and matching orderings. Becomes the recall oracle for #28.
4. **#28 Approximate-NN backend for the 16-D regime** — a third `NeighborSearch`
   (`ApproxNanoflannBackend`, built on the #24 k-NN path) using nanoflann approximate search
   (`SearchParameters.eps > 0`), with a recall knob and a bounded label-drift test.

## Tier 1 — Examples, datasets & validation

5. **#33 Shared dataset generators** — Python generators for `make_moons` / `make_blobs` / the
   "chameleon" 2-D sets emitting the `io.hpp` CSV contract. One source of truth feeding #25, #29,
   #32.
6. **#25 New examples on recognized datasets** — a 2-D example (noise handling, clean reachability
   plot) and a higher-D (16-D) example, each wired into `examples/CMakeLists.txt` like
   `optics_color`.
7. **#32 Cross-validate against scikit-learn OPTICS** — `tools/validate_sklearn.py` comparing
   labels (adjusted Rand index) and reachability (rank-correlation) within tolerance on the
   recognized datasets.

## Tier 2 — Docs, positioning & Python ergonomics

8. **#29 Honest comparison in README** — OPTICS vs k-means vs DBSCAN table + numbers (extend
   `examples/color_clustering/compare_kmeans.py` to also time a DBSCAN cut) + a committed
   side-by-side image.
9. **#30 First-time-user README** — a "run on your own data in 3 steps" quickstart, a
   reachability-plot reading guide (with image), and a parameter-selection guide
   (`min_pts`/`epsilon`/`chi`).
10. **#31 Reproducible Python env + one-command demo** — `requirements.txt`, a documented CSV
    workflow, and `scripts/demo.{ps1,sh}` that build + run the example on a bundled sample.
11. **#23 (stretch) pybind11 binding** — expose `compute_reachability_dists` + extraction for
    1/2/3/4-D clouds.

## Tier 3 — Release hardening

12. **#34** ASan/UBSan CI job + warnings-as-errors; Doxygen API docs; `CITATION.cff`
    (Ankerst et al., 1999 + this implementation).

---

## Sequencing

```
Tier 0:  #24 (land perf-dense) → #26 (perf scenarios) → #27 (equivalence) → #28 (approx-NN)
Tier 1:  #33 (datasets) → #25 (examples), #32 (sklearn validation)   ← share dataset infra
Tier 2:  #29, #30, #31 (docs + python) in parallel; #23 stretch
Tier 3:  #34 (CI + Doxygen + CITATION)  ← pre-tag polish
Release: verify checklist → tag v0.9.1
```

`#27` is **blocked-by** the perf scenario work where it shares fixtures; `#28` is **blocked-by**
`#27` (recall oracle) and the 16-D perf scenario. `#25`, `#29`, `#32` are **blocked-by** `#33`.

## Release checklist

- [ ] Newcomer path (clone → own-data plot) works on Windows + Linux, documented in README.
- [ ] README comparison table + honest k-means/DBSCAN positioning, backed by committed numbers.
- [ ] Nanoflann/Boost equivalence test green; approx-NN measured win + bounded recall test.
- [ ] `tools/validate_sklearn.py` passes within tolerance on the recognized datasets.
- [ ] Perf harness: dense-neighborhood + backend rows added; `perf/baseline.csv` refreshed; no
      regressions beyond tolerance.
- [ ] doctest suite + Boost-backend test green on MSVC, GCC, Clang.
- [ ] Release hardening (#34) landed: ASan/UBSan CI, Doxygen, `CITATION.cff`.
- [ ] `CHANGELOG.md` updated; version bumped to 0.9.1; tag `v0.9.1`.
