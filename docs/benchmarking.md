# Benchmarking: datasets, comparability, and findings

How we measure clustering **quality** and **speed**, against which datasets and engines, and the
caveats that make a comparison fair. Pairs with `tools/README.md` (how to run) and the 1.0.0
benchmark trackers (#53 CPU-speed comparisons, #54 quality harness).

## Datasets

| group | what | source | ground truth? |
|-------|------|--------|---------------|
| **Franti shape sets** | Aggregation, Compound, spiral, R15, jain, flame, D31 (2-D) | `cs.uef.fi/sipu/datasets` via `tools/fetch_datasets.py` → `data/franti/` | yes |
| **synthetic toys** | blobs, moons, varied, density (2-D, Euclidean) | `tools/datasets.py` generators | yes |
| **cosine blobs** | clusters along distinct directions, L2-normalized (8-D, 16-D) | `quality_benchmark.py:make_cosine_blobs` | yes |
| **RGB pixel clouds** | color-clustering images | `data/img_*.csv` | no (timing only) |

- The Franti sets are **third-party data with published results across many papers** (the same
  sets the FOPTICS paper reported on), so scores are comparable beyond this repo. **Please cite
  Franti et al.** if you publish results. They are *not* committed (`data/` is gitignored); run
  `tools/fetch_datasets.py` to (re)download.
- Everything talks through the CSV contract in `tools/README.md` (a coords file `x0,x1,…` + a
  `label` truth file).
- **Cosine blobs exist for sOPTICS.** sOPTICS is a cosine method, so it needs direction-based
  clusters to be evaluated fairly (see below); the Euclidean toys/Franti sets are the wrong metric
  for it.

## Comparability — what makes a comparison fair

These are the things that, if ignored, produce misleading numbers:

1. **Same generating distance (eps) — the big one.** OPTICS timing is dominated by neighborhood
   size, which eps controls. Our harness runs at the library's auto-estimated eps; it **emits that
   eps and passes the identical value to dbscan-R**. Skipping this made dbscan-R look ~50× slower on
   D31 (347 ms vs 7 ms) purely because it used a larger eps and therefore did more work. Always
   compare at equal eps.
2. **Cross-machine absolute times are not reproducible** (hardware differs). The reproducible,
   comparable quantity is the **ratio of engines on the same machine/datasets/params**. Use any
   published numbers to sanity-check your *setup*, not to match in absolute ms.
3. **Exclude I/O / startup** from the timed region on every side; time only the ordering.
4. **Like-for-like extraction.** We and dbscan-R both use Xi at the same `chi`/`xi`; the OPTICS
   *ordering* is the algorithm, the *extraction* is a separate, parameter-sensitive step.
5. **Tie-breaking differs.** We break reachability ties **low-index-first**; ELKI and mhahsler/dbscan
   break them **high-index-first**. So orderings are *not* bit-identical across engines even when both
   are exact — compare clusters (ARI/NMI/Rand) and timing, never raw orderings. (An opt-in
   high-index tie-break would let us match ELKI/dbscan exactly; not implemented.)
6. **Metric must match the method.** sOPTICS is cosine; scoring it on Euclidean layouts is a metric
   mismatch, not a defect. scikit-learn `OPTICS(cluster_method="xi")` is sensitive to `xi` and can
   score low at defaults — it is a reference, not a target.

## Engines compared

| engine | how | status |
|--------|-----|--------|
| **ours** (OPTICS, sOPTICS) | `optics_quality_compare` (C++) | built here |
| **scikit-learn** (OPTICS xi, HDBSCAN) | `quality_benchmark.py` (Python) | runs here |
| **mhahsler/dbscan** (R; ANN kd-tree OPTICS) | `run_dbscan_r.R` via `quality_benchmark.py` | runs here (needs R + `dbscan`) |
| **ELKI** (Java; OPTICSXi/FastOPTICS/HDBSCAN\*) | manual, same CSVs | not run (no java); `tools/README.md` |
| **NinhPham/sDbscan** (C++; sOPTICS competitor) | manual, build from source | not run; `tools/README.md` |

R-sandbox note: if a sandboxed R install hides the package from a subprocess, set `R_LIBS_USER` to
the library holding `dbscan` (`Rscript -e "cat(dirname(find.package('dbscan')))"`). The harness
probes availability and skips the column cleanly otherwise.

## Findings (representative, this machine — ratios, not absolutes)

- **Speed.** At equal eps, **ours-OPTICS and dbscan-R are neck-and-neck** (e.g. D31 5 vs 7 ms,
  aggregation 1 vs 2 ms), both **~100–1000× faster than scikit-learn OPTICS** (D31 1557 ms;
  16-D cosine 7.5 s). Our heap-based seed queue edges out dbscan's linear-scan seed list, most
  visibly at the larger D31.
- **sOPTICS.** Matches the exact methods in its cosine regime (8-D cosine blobs: identical ARI
  ≈0.78) and is honestly lower on Euclidean layouts (metric mismatch). It crosses over to faster
  than exact OPTICS only at larger n / higher density (see `optics_soptics_compare`).
- **Quality parity.** At the **knee** eps, ours-OPTICS matches or beats dbscan-R on most Franti sets
  (e.g. R15 0.95 vs 0.91, aggregation 0.97 vs 0.54); HDBSCAN is still strongest on a few.
- **Generating-distance estimator matters a lot on clustered data (#57).** The uniform-density
  `epsilon_estimation` over-shoots on clustered data and over-smooths the reachability, so Xi
  under-segments — most starkly on R15 (ARI **0.43**). The k-distance-knee estimator
  (`epsilon_estimation_knee`, #41) lands near the within-cluster scale and recovers it (R15 ARI
  **0.95**), and also helps compound, spiral, and density. The quality harness now defaults to
  `--eps knee` (passed to dbscan-R too, so timing stays a fair same-eps comparison). It was NOT the
  Xi steep-area logic — a decoupled `min_cluster_size` (added for ELKI parity) had no effect on R15.

## TODOs

- ~~Consider the knee estimator as the auto-eps default~~ **DONE** (#49/#57): auto-epsilon now
  defaults to `epsilon_estimation_knee` across `compute_reachability_dists` / `cluster_threshold` /
  `extract_xi` (falls back to uniform for non-`KnnCoreDist` backends and degenerate inputs). Pass an
  explicit `epsilon` for the uniform behavior.
- **sOPTICS 16-D quality dip** — cos-blobs-16d ARI 0.57 vs ~0.82 for the exact methods; check
  CEOs params (D / k / m) and curse-of-dimensionality.
- **Reproduce the dbscan JSS benchmark** (replication script) to validate our setup against
  *published* numbers, not just same-machine ratios.
- **ELKI + sDbscan columns** — need their runtimes; ELKI especially is the FastOPTICS/sOPTICS parity
  reference (#53).
- **FHT structured projections** for sOPTICS — would lower its small-scale projection overhead.
