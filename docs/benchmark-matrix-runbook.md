# Runbook — executing the 1.0.0 benchmark matrix (B2) on the reference machine

This is the **operational guide** for running the study (issue #59). It assumes the infrastructure is
built (it is — see [`ROADMAP-1.0.0-execution.md`](ROADMAP-1.0.0-execution.md) Part B); here we just
*run* it. For the study's *design* (axes, decisions D1–D5, fairness rules) see
[`ROADMAP-1.0.0-benchmark-matrix.md`](ROADMAP-1.0.0-benchmark-matrix.md); for methodology see
[`benchmarking.md`](benchmarking.md) and [`../perf/README.md`](../perf/README.md).

The whole study is four scripts on a shared CSV contract:

```
gen_dataset.py  →  optics_matrix (C++) + sk_engine.py     →  run_matrix.py  →  analyze_matrix.py
  (one dataset)     (one engine, one cell, same CSV)          (orchestrate)      (D1–D5 tables)
```

Everything appends to one growing, checkpointed CSV, so the run is **kill-safe**: stop any time, re-run
the same command with `--resume`, and it continues.

---

## 1. Reference machine & prerequisites

Pick **one quiet machine** and run the whole study there — only same-machine engine *ratios* are
reproducible (cross-machine absolute times are not; `benchmarking.md` §Comparability). Record its
spec; it becomes part of the citable reference.

- **OS / compiler:** any of the supported toolchains (MSVC 2022, GCC 10+, Clang 13+), C++20, CMake ≥ 3.21.
- **Python:** 3.10+ with `numpy` and `scikit-learn` (`pip install -r requirements.txt`). scikit-learn is
  required — it provides both the competitor engines *and* the ARI/NMI/Rand scorer.
- **RAM:** ≥ 16 GB for the default tiers (capped at n = 1e6). The 1e7 level is not in the default tiers.
- **(optional) R + `dbscan`** for the dbscan-R column (auto-probed; skipped cleanly if absent — see the
  R-sandbox note in `tools/README.md`).
- **Quiet the box:** close other heavy processes. Timing is the product here; a noisy neighbour skews it.

> ELKI and NinhPham/sDbscan are **deferred past 1.0.0** to a Docker repro env and are not part of this
> run (decided 2026-06-08).

---

## 2. One-time setup

```sh
# build the C++ per-cell executor in Release (this is what runs ours-*)
cmake --preset msvc                                 # or: linux-gcc / linux-clang
cmake --build --preset msvc --target optics_matrix
```

Pin the worker-thread count so numbers are reproducible (the harness default is already 4; set it
explicitly and keep it fixed for the whole study — it is recorded in every result row):

```sh
# Windows PowerShell
$env:OPTICS_BENCH_THREADS = "4"
# Linux/macOS
export OPTICS_BENCH_THREADS=4
```

---

## 3. The run sequence

Run from the repo root. Every step appends to the **same** `results/matrix.csv`; `--resume` skips
already-recorded `(cell, engine, config, rep)` groups, so the steps compose and the run is restartable.

```sh
# 0) pre-flight: every engine must recover clean, well-separated clusters (ARI ~ 1) before we time
#    anything. Exits non-zero on failure -- do not proceed past a red gate.
python tools/correctness_gate.py

# 1) pilot smoke (~1-2 min): proves the whole chain end-to-end on this machine
python tools/run_matrix.py --tier pilot --out results/matrix.csv --reps 1

# 2) Tier A -- scaling spine (n full at d in {3,16}). A tighter budget auto-skips the very slow
#    scikit-learn OPTICS at large n (it is ~O(n^2)); its headline-speedup role is captured at small n.
python tools/run_matrix.py --tier scaling --out results/matrix.csv --reps 3 --budget-s 120 --resume

# 3) Tier B -- dimensionality spine (d full at n in {1e4,1e5})
python tools/run_matrix.py --tier dim --out results/matrix.csv --reps 3 --budget-s 120 --resume

# 4) randomized-engine variance: >= 3 seeds for sOPTICS / sHDBSCAN on a pinned subset (design §9.8).
#    --reps 5 runs 5 seeds; --refresh re-runs just these engines and supersedes their earlier rows.
python tools/run_matrix.py --tier pilot --engines ours-soptics ours-shdbscan \
       --reps 5 --out results/matrix.csv --refresh

# 5) analyze -> D1-D5 decision tables, speedup-vs-sklearn, quality table (Markdown report)
python tools/analyze_matrix.py results/matrix.csv --out results/report.md
```

**Budget knob.** `--budget-s` is the per-run feasibility cap: a cell whose *predicted* cost exceeds it
is skipped and **logged** (status `skipped_budget`), never silently dropped and never left to hang
(scikit-learn runs in a subprocess and is hard-killed on timeout). Raise it (e.g. `--budget-s 600`,
the default) to include slower cells; lower it for a faster first pass.

---

## 4. Estimated cost (upper bounds, threads = 4, reps = 3, this machine class)

From the per-engine cost model (`run_matrix.predicted_seconds`); conservative, real time is usually
less. The Phase-1 pilot refines these on *your* box — record the actuals.

| tier | cells | engine-runs | gated skips | ~wall-clock (budget 600s) |
|------|-------|-------------|-------------|---------------------------|
| pilot | 4 | ~84 | 0 | ~1–2 min |
| scaling | 16 | ~214 | ~10 | ~40 min |
| dim | 22 | ~308 | 0 | ~140 min |

With `--budget-s 120` the scikit-learn-OPTICS cells above ~5e4 auto-skip, cutting the scaling/dim
wall-clock substantially (those few cells dominate the upper bound). Plan for **~1–3 h** for a full
default pass; it is checkpointed, so it can span sessions.

---

## 5. What the study currently resolves (and what needs more axes)

The shipped tiers resolve the **crossover and headline** decisions and the quality baseline:

- **D2 (sOPTICS vs OPTICS)** and **D5 (HDBSCAN\* vs sHDBSCAN)** — the `n`-spine gives the crossover.
- **Speedup vs scikit-learn OPTICS** and **ours-HDBSCAN vs scikit-learn-HDBSCAN parity** — every cell.
- **Quality** (ARI/NMI/Rand vs ground truth) across `d`, density, and shape.

`analyze_matrix.py` is **honest about what it can't yet decide**: **D1** (backend by dimensionality)
needs the backend axis (exact / eps-approx / Boost / HNSW), **D3** (Precompute vs OnDemand) needs the
mode axis, **D4** (uniform vs knee ε) needs the eps axis. `optics_matrix` already accepts `--eps` and
`--mode`; wiring those as sweep axes in `run_matrix.py` (a new tier or a config dimension) is the
documented next step before D1/D3/D4 can be closed. The analysis prints exactly which axis each
unresolved decision is waiting on.

---

## 6. Capture the provenance (makes it a *citable* reference)

Each result row already records the **git commit, thread count, and timestamp**. Add the rest of the
environment to the report header by hand (design §5) — without it the "single reference" is not
citable:

- CPU model + core count, RAM, OS version, compiler + version, Python + scikit-learn versions.
- Confirm the working tree was **clean** at the recorded commit (`git status`), so the binary matches.

```sh
git rev-parse HEAD && git status --porcelain     # commit + clean check
python -c "import sklearn, numpy, sys; print(sys.version, sklearn.__version__, numpy.__version__)"
```

---

## 7. Resuming, and updating the matrix later

- **Resume a killed run:** re-run the same command with `--resume`. Completed groups are skipped;
  dataset generation is skipped when nothing is left for a cell.
- **Update after an infrastructure change** (new backend, tuned default, new algorithm): re-run only
  the affected engine and let the analysis take the newest rows —

  ```sh
  python tools/run_matrix.py --out results/matrix.csv --engines ours-optics --refresh
  python tools/analyze_matrix.py results/matrix.csv      # keeps the latest row per measurement
  ```

  To add an engine, append one tuple to the registry in `run_matrix.py` (`OURS_ENGINES` / `SK_ENGINES`)
  and, for an `ours-*` engine, an `--algo` branch in `optics_matrix.cpp`. See
  [`ROADMAP-1.0.0-execution.md`](ROADMAP-1.0.0-execution.md) §B1.7.

---

## 8. After the run → 1.0.0

Per the design's phasing: resolve D1–D5 from the report → set the library's data-dependent defaults →
re-run a confirmation subset (`--refresh` the changed engines) → fold the tables/figures into
`benchmarking.md` + the README → **freeze the public API (#49)** → tag **1.0.0**.
