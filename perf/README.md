# Performance tracking

`optics_perf` (built from `test/Benchmark/perf.cpp`, nanobench) measures the hot paths and a
few end-to-end ordering scenarios, and writes `optics_perf.csv`. Use it to quantify the impact
of the Tier-0 performance work (see `docs/ROADMAP-0.9.md`).

## Running

```sh
cmake --build build --config Release --target optics_perf
./build/test/Release/optics_perf      # writes optics_perf.csv in the working dir
```

## `baseline.csv`

Committed reference, captured **before** task #12 (`compute_core_dist` optimization).

- Machine: 22-thread desktop (Windows, MSVC 19.44, Release).
- Headline metric for #12: **`core_dist 3D double (30k)`** (per-call ns, `elapsed` column ÷ batch).

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
