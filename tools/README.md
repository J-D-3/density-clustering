# Python tools

Optional glue around the C++ library: generate data, cluster it (via the
`cluster_csv` example), and visualize / compare / validate the result. None of it is
required to build or use the library — install the deps with
`pip install -r ../requirements.txt`.

For benchmarking methodology — which datasets, the comparability caveats (same-eps, tie-breaking,
metric matching), engine status, and findings/TODOs — see [`../docs/benchmarking.md`](../docs/benchmarking.md).

## The CSV contract

Everything talks through two dimension-agnostic CSV shapes written by
`optics::io` (see `include/optics/io.hpp`):

| file | header | rows | written by |
|------|--------|------|------------|
| **points** | `x0,x1,...,x{Dim-1},cluster_id` | one point per row; `cluster_id` = -1 for noise | `export_points_csv` (the examples) |
| **reachability** | `order_index,point_index,reachability` | the OPTICS ordering; `reachability` = -1 if unreached | `export_reachability_csv` |

A **coordinates input** file is just the points file without the label column: a
`x0,x1,...` header and one numeric row per point (`tools/datasets.py` writes these,
and `cluster_csv` reads them).

## Scripts

| script | what it does |
|--------|--------------|
| `datasets.py` | Generate reproducible 2-D datasets (`blobs` / `moons` / `varied`) as a coordinates CSV plus a ground-truth labels CSV. |
| `visualize.py` | Render a points CSV (2-D/3-D scatter, PCA for higher-D) and, optionally, the reachability plot. `--out plot.png` saves headlessly. |
| `compare_algorithms.py` | Side-by-side figure: OPTICS (this library) vs scikit-learn k-means and DBSCAN. |
| `validate_sklearn.py` | Cross-check our reachability + labels against `sklearn.cluster.OPTICS` (Spearman + ARI). |
| `timing_compare.py` | Time the internal backends (nanoflann / approx / Boost) vs `sklearn.cluster.OPTICS` across cases; needs the `optics_backend_compare` harness. |
| `timing_images.py` | Color-clustering runtime on RGB images: internal backends vs scikit-learn OPTICS / DBSCAN / KMeans on the same sampled pixel cloud. |
| `quality_benchmark.py` | Score OPTICS + sOPTICS (ours), scikit-learn OPTICS + HDBSCAN, **and mhahsler/dbscan (R)** against ground truth (ARI / NMI / Rand) + a timing table; needs the `optics_quality_compare` harness (and, for the dbscan-R column, R + the `dbscan` package). |
| `hdbscan_benchmark.py` | Cross-check our HDBSCAN\* + sHDBSCAN against `sklearn.cluster.HDBSCAN` and ground truth (ARI / NMI / Rand) plus a direct **ours-vs-sklearn label-agreement** column (the correctness signal for #52); needs the `hdbscan_compare` harness. `min_samples` is self-inclusive in both and passed identically. |
| `fetch_datasets.py` | Download Franti's clustering benchmark "shape sets" (Aggregation, Compound, spiral, R15, jain, flame, D31) into `data/franti/` as coords + ground-truth CSVs — third-party data with published results (please cite Franti et al.). |
| `run_dbscan_r.R` | Run mhahsler/dbscan's OPTICS+Xi on a coords CSV → predicted labels + timing (invoked by `quality_benchmark.py`; needs R + `dbscan`). |
| `gen_dataset.py` | **(matrix)** Parametrized generator: `(n, d, k, density, noise, shape, seed)` → coords + truth CSV. Dimension-stable separation-ratio density (dense/sparse/mixed), d-D blobs, and non-spherical shapes (moons/spiral) embedded by a random rotation. |
| `run_matrix.py` | **(matrix)** Orchestrator: expand a tier → cells, generate each once, run every engine on the same CSV, score ARI/NMI/Rand, append a tidy long-format CSV. Checkpoint/resume, feasibility-gated, full provenance. Needs the `optics_matrix` harness. |
| `correctness_gate.py` | **(matrix)** Pre-flight: assert every engine recovers clean, well-separated clusters (ARI≈1) before any timing — catches a broken config. Exits non-zero on failure (CI/run gate). |
| `analyze_matrix.py` | **(matrix)** Turn the tidy CSV into the D1–D5 decision tables, the speedup-vs-sklearn-OPTICS table, and a quality table (Markdown); honestly flags decisions that still need an unswept axis. |
| `sk_engine.py` | **(matrix)** scikit-learn per-cell worker (OPTICS/HDBSCAN/DBSCAN/KMeans) invoked by `run_matrix.py` as a subprocess so it can be hard-timeout-killed (a sklearn `fit` can't be interrupted in-process). Not run directly. |

```sh
cmake --build --preset msvc --target optics_quality_compare
python tools/fetch_datasets.py            # Franti shape sets -> data/franti/
python tools/quality_benchmark.py --exe build/test/Release/optics_quality_compare
```

Note: **sOPTICS is a cosine method** — it scores well on the `cos-blobs-*` (direction-based)
rows and lower on the Euclidean 2-D toys; that is the metric, not a defect. `sk-OPTICS` uses
Xi extraction, which is parameter-sensitive (so it can score low at the default `xi`).

## The 1.0.0 benchmark matrix (#59)

`gen_dataset.py` → `run_matrix.py` → `analyze_matrix.py` are the consolidated reference study
(design: [`../docs/ROADMAP-1.0.0-benchmark-matrix.md`](../docs/ROADMAP-1.0.0-benchmark-matrix.md);
execution: [`../docs/ROADMAP-1.0.0-execution.md`](../docs/ROADMAP-1.0.0-execution.md)). The C++
per-cell executor is `optics_matrix` (one algorithm per call: `optics`/`soptics`/`hdbscan`/`shdbscan`,
dim spine {1..128}). Every engine reads the *same* generated CSV — that shared input is what makes the
comparison fair.

```sh
cmake --build --preset msvc --target optics_matrix
python tools/correctness_gate.py                                   # pre-flight: every engine recovers clean blobs
python tools/run_matrix.py --tier pilot --out results/matrix.csv   # add --resume to continue a killed run
python tools/analyze_matrix.py results/matrix.csv --out results/report.md
```

Tiers: `pilot` (tiny end-to-end smoke), `scaling` (n-spine), `dim` (d-spine) — `--list-tiers` to
see cell counts. ELKI / NinhPham-sDbscan are **deferred past 1.0.0** to a Docker repro env (see
below). `results/` and `data/` are gitignored (reproducible from these scripts).

**Updating the matrix after an infrastructure change** (new backend, tuned default, new algorithm):
the run is built to be re-run in part, not wholesale. Each row records the git commit; to refresh
just the affected engine after changing its code, re-run only it and let the analysis take the
newest rows:

```sh
python tools/run_matrix.py --out results/matrix.csv --engines ours-optics --refresh
python tools/analyze_matrix.py results/matrix.csv      # keeps the latest row per measurement
```

To add an engine/algorithm, append one tuple to the engine registry in `run_matrix.py`
(`OURS_ENGINES` / `SK_ENGINES`) — and, for an `ours-*` engine, an `--algo` branch in
`optics_matrix.cpp`. Gating, scoring, the tidy CSV, and the analysis are unchanged.

## Comparing against mhahsler/dbscan (R)

`quality_benchmark.py` adds a `dbscan-R` column when R and the `dbscan` package are available
(auto-detected; otherwise the column is skipped with a note). Our OPTICS and dbscan-R are both
exact-Euclidean and run at the **same generating distance** — our harness emits the `eps` it
used and passes it to dbscan-R — so the timing is comparable; tie-breaking differs (we
low-index-first, dbscan/ELKI high-index-first), so compare clusters (ARI/NMI/Rand), not
bit-identical orderings.

```sh
# install R (e.g. `winget install RProject.R`), then the package:
Rscript -e "install.packages('dbscan', repos='https://cloud.r-project.org')"
```

If R is in a sandboxed/redirected profile and the package isn't found from a subprocess, point
`R_LIBS_USER` at the library holding `dbscan`
(`Rscript -e "cat(dirname(find.package('dbscan')))"`).

## Other external engines (not bundled)

ELKI (Java) and NinhPham/sDbscan (C++) are not invoked automatically (licenses prevent
vendoring; they need their own runtimes). Run them on the same `data/` CSVs and add their
timings by hand:

- **ELKI** (OPTICSXi / FastOPTICS / HDBSCAN\*): `java -jar elki.jar KDDCLIApplication
  -algorithm clustering.optics.OPTICSXi -dbc.in data/qb_franti_aggregation.csv ...`.
- **NinhPham/sDbscan** (C++; the direct sOPTICS competitor): build from source, run on the
  same CSVs. See `documentation/references.md` for links and licensing notes.

## End-to-end

```sh
python tools/datasets.py --name varied --n 1500 --out data/varied.csv
build/examples/Release/cluster_csv data/varied.csv data/varied 10 -1 2.5 0.01
python tools/visualize.py --points data/varied_points.csv --reach data/varied_reach.csv
```

Or run it all at once with [`scripts/demo.ps1`](../scripts/demo.ps1) (Windows) /
[`scripts/demo.sh`](../scripts/demo.sh) (Linux/macOS).
