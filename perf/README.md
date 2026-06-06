# Performance tracking

`optics_perf` (built from `test/Benchmark/perf.cpp`, nanobench) measures the hot paths and a
few end-to-end ordering scenarios, and writes `optics_perf.csv`. Use it to quantify the impact
of the Tier-0 performance work (see `docs/ROADMAP-0.9.md`).

## Running

```sh
cmake --build build --config Release --target optics_perf
./build/test/Release/optics_perf      # writes optics_perf.csv in the working dir
```

**Threads.** All timing harnesses (`optics_perf`, `optics_benchmark`, `optics_scale`) default
to **4 worker threads** so numbers are reproducible and comparable across machines instead of
scaling with the dev box's core count. Override with the `OPTICS_BENCH_THREADS` environment
variable (e.g. `OPTICS_BENCH_THREADS=16`). For the memory-bound precompute phase, 4 threads is
often *faster* than saturating all cores.

## `baseline.csv`

Committed reference for the current line of development (refreshed for v0.9.1).

- Machine: 22-thread desktop (Windows, MSVC 19.44, Release).
- Headline metric for #12: **`core_dist 3D double (30k)`** (per-call ns, `elapsed` column ÷ batch).

### v0.9.1 scenarios

- **`dense 3D 30k core-dist {scan,knn}`** — a few very dense blobs (flat-color-like), so
  each point's eps-neighborhood is huge. The Knn core-distance (issue #24) avoids scanning
  the neighborhood and is faster here (~2.9 s → ~2.4 s, ≈16% at 4 threads); the gap widens as
  neighborhoods grow.
- **`backend 16D 8k nanoflann {exact,approx}`** (plus `boost rtree` when built with
  `-DOPTICS_ENABLE_BOOST_RTREE=ON`) — the same 16-D cloud across backends, comparing the
  exact KD-tree, the eps-approximate backend (issue #28), and Boost's R*-tree.

## How to compare (important)

This is a noisy multi-core desktop: run-to-run variation on the microbenchmarks can reach
~30%, larger than the intra-run `error %`. So **do not** compare a fresh run against a
days-old `baseline.csv` and trust small deltas. Instead, for each essential change:

1. Build + run `optics_perf` on the parent commit, save the CSV.
2. Build + run on the change, save the CSV.
3. Compare the two **back-to-back** runs; treat only changes clearly above the run-to-run
   noise (rule of thumb: > ~15–20% on the microbenchmarks) as signal.

Refresh `baseline.csv` after each landed Tier-0 change so it tracks the current best.
CI may run `optics_perf` informationally, but timings are **not** a gate (runner hardware
varies).

## Large-scale validation (`optics_scale`)

`optics_scale [n]` times the 3D color-space-style workload (uniform cloud, min_pts=16) at
1e6–1e7 points. The figures below are indicative, captured on a 22-thread desktop (Release) at
full hardware concurrency; the harness now defaults to 4 threads (override with
`OPTICS_BENCH_THREADS`), so Precompute timings on the same box will differ:

| workload      | Precompute       | OnDemand (x1) |
|---------------|------------------|---------------|
| 3D float  1e6 | ~5.8 s           | ~7.8 s        |
| 3D double 1e6 | ~6.7 s           | ~9.3 s        |
| 3D float  1e7 | ~44.0 s          | ~52.0 s       |
| 3D double 1e7 | ~45.7 s          | ~55.7 s       |

Both modes order the full cloud; Precompute did not OOM at 1e7 on the desktop. Note the modest
Precompute-vs-OnDemand gap at 1e7: the sequential ordering loop is a large fraction at this
scale, so parallelizing only the query phase is Amdahl-limited. Confirms the Tier-0 finding that
OPTICS is **query-bound** — future speedups live in the query path (backend, approximate-NN),
not the surrounding bookkeeping.
