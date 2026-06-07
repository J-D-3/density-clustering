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
| `quality_benchmark.py` | Score OPTICS + sOPTICS (ours) and scikit-learn OPTICS + HDBSCAN against ground truth (ARI / NMI / Rand), plus an ours-vs-sklearn timing table; needs the `optics_quality_compare` harness. |

```sh
cmake --build --preset msvc --target optics_quality_compare
python tools/quality_benchmark.py --exe build/test/Release/optics_quality_compare
```

Note: **sOPTICS is a cosine method** — it scores well on the `cos-blobs-*` (direction-based)
rows and lower on the Euclidean 2-D toys; that is the metric, not a defect. `sk-OPTICS` uses
Xi extraction, which is parameter-sensitive (so it can score low at the default `xi`).

## Comparing against external engines (#53)

The strongest CPU competitors are not bundled and are **not invoked** by the scripts above
(their licenses prevent vendoring, and they need their own runtimes). To add them to a timing
comparison, install and run them on the same coordinate CSVs the harnesses write to `data/`,
then drop their per-dataset milliseconds into the tables by hand or extend the scripts:

- **ELKI** (Java; OPTICSXi / FastOPTICS / HDBSCAN\*): `java -jar elki.jar KDDCLIApplication
  -algorithm clustering.optics.OPTICSXi -dbc.in data/qb_blobs-2d.csv ...`.
- **mhahsler/dbscan** (R; ANN kd-tree OPTICS): `Rscript -e 'optics(read.csv(...), eps=..., minPts=...)'`.
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
