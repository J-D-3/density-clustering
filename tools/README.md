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

## End-to-end

```sh
python tools/datasets.py --name varied --n 1500 --out data/varied.csv
build/examples/Release/cluster_csv data/varied.csv data/varied 10 -1 2.5 0.01
python tools/visualize.py --points data/varied_points.csv --reach data/varied_reach.csv
```

Or run it all at once with [`scripts/demo.ps1`](../scripts/demo.ps1) (Windows) /
[`scripts/demo.sh`](../scripts/demo.sh) (Linux/macOS).
