# Plan — closing 1.0.0 and executing the benchmark matrix

**Status as of 2026-06-08.** This is the *execution* companion to
[`ROADMAP-1.0.0-benchmark-matrix.md`](ROADMAP-1.0.0-benchmark-matrix.md) (the study *design*) and
[`ROADMAP-post-0.9.1.md`](ROADMAP-post-0.9.1.md) (the backlog). It tracks the last loose ends before
the **1.0.0** tag and the concrete steps to run the one reference study.

New developers: read `benchmarking.md` and `perf/README.md` first — they hold the methodology and the
fairness contract this plan obeys.

---

## 0. Where 1.0.0 actually stands

Most of the 1.0.0 feature work is merged to `master` (the `[Unreleased]` block in `CHANGELOG.md`).
The remaining work is small, but the **GitHub issue states had drifted** from the code — several
features were merged while their issues stayed open. Ground truth:

| Issue | Feature | On `master`? | Action |
|---|---|---|---|
| #46 | Weighted / unique-point OPTICS | ✅ | closed |
| #47 | HNSW high-D backend | ✅ (PR #64) | **close** |
| #50 | sOPTICS | ✅ | closed |
| #51 | Non-Euclidean metrics (L2 / L1) | ✅ | **close** + file χ²/JS follow-up |
| #52 | **HDBSCAN\* + sHDBSCAN** | ✅ (PR #65) | **close** |
| #54 | Quality harness | ✅ | closed |
| #55 | Reuse search distances | ✅ (PR #62) | **close** |
| #58 | sOPTICS refinements (FHT + 16-D + data-scaled eps) | ✅ | **close** |
| #53 | CPU competitor comparisons | ⚠️ partial | finish (A4) |
| #48 | Perf-regression CI gate | ❌ | do (A5) |
| #49 | API-stability + semver freeze | ❌ | do (A6, **last** — gates the tag) |
| #59 | Benchmark matrix | design done, **not run** | **re-open / re-track** (B) |

HDBSCAN\* and sHDBSCAN landed via PR #65 (`include/optics/hdbscan.hpp`, 886 lines), rebased onto
master so the branch's alternative #58 16-D fix (`9ba1615`) was dropped in favour of master's Xi
fix — both the Xi fix and the data-scaled auto-eps survive on master. CHANGELOG covers #52 and #58.

---

## Part A — close the remaining gaps

**A1 — Merge HDBSCAN\* / sHDBSCAN (#52). ✅ DONE** (PR #65; verified `hdbscan.hpp` on master, build +
ctest green). Was the only blocker for the matrix.

**A2 — Backfill the #58 CHANGELOG entry. ✅ DONE** (FHT structured projections + data-scaled auto-eps
both documented in `[Unreleased]`).

**A3 — Issue hygiene.** Close #47, #51, #52, #55, #58 (all merged). For #51, file a fresh issue for
the deferred χ² / Jensen-Shannon metrics so the deferral is tracked, not lost. Re-open #59 (or file a
new "execute benchmark matrix" issue) so the *run* is tracked — the design landing is not the study
running. Prune the merged feature branches (`feat/hdbscan-52`, `feat/hnsw-backend`,
`feat/soptics-refinements-58`) locally and on origin.

**A4 — Finish competitor comparisons (#53).** Done: scikit-learn + mhahsler/dbscan (R), at identical
eps. Remaining: **ELKI** (needs a JVM — absent in the dev env) and **NinhPham/sDbscan** (build from
source). Both are environment-fragile; fold their adapters into the matrix's Docker repro env (§8.4 of
the design) rather than fighting the local sandbox. If 1.0.0 ships without them, record
"ELKI / sDbscan deferred" explicitly in `benchmarking.md`.

**A5 — Perf-regression CI gate (#48).** `optics_perf` is informational today. Add a tolerance-based
**back-to-back** compare step on the key microbenchmarks (per `perf/README.md` — compare two fresh
runs, *not* a stale committed baseline; microbench noise reaches ~30%). Also add an
`OPTICS_ENABLE_HNSW=ON` CI job — HNSW ships but no CI job exercises it.

**A6 — API-stability + semver freeze (#49). [gates the tag]** Freeze the public `optics::` surface,
document `detail::` as explicitly unstable, and record the pre-1.0 behaviour changes already made
(OnDemand default flip in 0.9.1; knee-eps default in #57). Do this **after** the matrix picks the
data-dependent defaults (Part B Phase 3) so the frozen API reflects them.

---

## Part B — execute the benchmark matrix (#59)

The design (`ROADMAP-1.0.0-benchmark-matrix.md`) is sound: tiers A–E, decisions D1–D4, fairness
contract §9. Two amendments and the infrastructure build stand between it and a run.

### B0 — Amend the design to cover HDBSCAN + sHDBSCAN (the gap)

The design predates the HDBSCAN/sHDBSCAN work; §4 lists exact HDBSCAN\* in the *quality* tier only.
Update it to:

- Add **sHDBSCAN** as an engine in §4 alongside sOPTICS (cosine; vary `n_projections` / `k` / `m`).
- Put exact HDBSCAN\* in the **Tier-A scaling spine** and **Tier-B dimensionality spine**, not just
  quality — its dense-Prim MST is **O(n²)** in time / O(n) memory, exactly the scaling story those
  spines exist to measure (and its scaling wall is the motivation for sHDBSCAN).
- Add decision **D5 — exact HDBSCAN\* vs sHDBSCAN crossover** (`n × density × metric` surface,
  mirroring D2 for OPTICS/sOPTICS), with a Tier-D probe.
- Fold the merged harnesses into §7: `test/Benchmark/hdbscan_compare.cpp` (per-cell HDBSCAN executor)
  and `tools/hdbscan_benchmark.py` (sklearn-HDBSCAN cross-check — already scores ARI/NMI/Rand + an
  ours-vs-sklearn agreement column; our exact HDBSCAN matched scikit-learn at ARI 0.99–1.00 across 13
  datasets). Add scikit-learn-contrib/`hdbscan` and TutteInstitute/`fast_hdbscan` as references.

### B1 — Build the §8 infrastructure

1. **✅ DONE — `tools/gen_dataset.py`** (`(n, d, k, density, noise, shape, seed)` → coords CSV +
   truth CSV). Built in **Python**, not as a `testdata.hpp` extension: the orchestrator is Python and
   every engine (ours via `csv_points.hpp`, sklearn, dbscan-R, ELKI) consumes the *same* on-disk CSV,
   which is what makes the comparison fair (§7 / §9); `testdata.hpp` stays for in-process C++ unit
   fixtures. Implements the §2 separation-ratio density (`rho = sep/sigma`; dense≈2, sparse≈10, mixed
   = varied σ + varied populations + noise — verified exact), exact d-D min-separation blob placement,
   and non-spherical shapes (moons/spiral) as thin manifolds embedded by a random orthonormal rotation
   (so they stay recoverable in any d and aren't axis-aligned). Sanity-checked: sklearn-HDBSCAN ARI
   0.78–0.95 vs truth across regimes at d=2..16. Reuses `datasets.write_csv`.
2. **`optics_matrix` exe** — one binary with explicit `Dim` instantiations for
   {1,2,3,4,6,8,12,16,32,64,128} dispatched by a runtime `switch(dim)` (watch binary size).
3. **`tools/run_matrix.py`** — expand tier specs → cells; feasibility-gate + timeout (§3b: predict
   cost, cap, **log every skip** — never silently truncate); invoke C++ harnesses + adapters; append
   one tidy long-format CSV; **checkpoint / resume**; record full provenance (§5).
4. **`tools/analyze_matrix.py`** — D1–D5 decision tables, Pareto fronts, speedup-vs-sklearn figures.
5. **Adapters + Docker repro env** — sklearn ✓, dbscan-R ✓, ELKI (new), NinhPham/sDbscan (new); all
   containerized so the "single reference" reproduces off the dev box (R-sandbox / Java / build
   fragility is the main operational risk).
6. **Correctness gate** — before any timing, ARI ≈ 1 vs the brute-force reference on clean blobs per
   small cell, so we never benchmark a broken config.

### B2 — Run (design §10 phasing)

- **Phase 0 — prerequisites:** A1 ✅; B1 infra green; correctness gate passing.
- **Phase 1 — pilot:** small-n cost models; finalize feasibility caps; publish cell count +
  wall-clock budget into the design doc §3c.
- **Phase 2 — full run:** Tiers A–E + Latin-hypercube fill, checkpointed, in-container, reporting ≥3
  seeds for the randomized engines (sOPTICS / sHDBSCAN — §9.8).
- **Phase 3 — decide & implement:** resolve D1–D5 → concrete library defaults → re-run a confirmation
  subset to verify the new defaults beat the old.
- **Phase 4 — publish & tag:** fold tables/figures into `benchmarking.md` + README; then **A6 (API
  freeze)**; then tag **1.0.0**.

**Critical path:** A1 ✅ → B0 (amend design) → B1 (infra) → B2 P0–P3 → A6 (freeze) → tag.
A3 / A4 / A5 run in parallel and don't block the matrix.

**Exit criteria (from the design):** every D1–D5 rule backed by data (or recorded "no significant
difference"); no silently-skipped cells (all gated skips logged); reference tables reproducible from
`run_matrix.py` on a clean container.
