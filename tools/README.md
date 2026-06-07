# Python tools

Optional glue around the C++ library: generate data, cluster it (via the
`cluster_csv` example), and visualize / compare / validate the result. None of it is
required to build or use the library — install the deps with
`pip install -r ../requirements.txt`.

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
| `fetch_datasets.py` | Download Franti's clustering benchmark "shape sets" (Aggregation, Compound, spiral, R15, jain, flame, D31) into `data/franti/` as coords + ground-truth CSVs — third-party data with published results (please cite Franti et al.). |
| `run_dbscan_r.R` | Run mhahsler/dbscan's OPTICS+Xi on a coords CSV → predicted labels + timing (invoked by `quality_benchmark.py`; needs R + `dbscan`). |

```sh
cmake --build --preset msvc --target optics_quality_compare
python tools/fetch_datasets.py            # Franti shape sets -> data/franti/
python tools/quality_benchmark.py --exe build/test/Release/optics_quality_compare
```

Note: **sOPTICS is a cosine method** — it scores well on the `cos-blobs-*` (direction-based)
rows and lower on the Euclidean 2-D toys; that is the metric, not a defect. `sk-OPTICS` uses
Xi extraction, which is parameter-sensitive (so it can score low at the default `xi`).

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
