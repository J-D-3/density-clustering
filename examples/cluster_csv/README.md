# cluster_csv — cluster your own point cloud

A small end-to-end example: read a coordinates CSV, run OPTICS, and write CSVs you
can plot. Works for **2-, 3-, 4-, and 16-D** data (dimension is read from the
header). This is the fastest way to try OPTICS on your own data.

## 1. Get some data

Either point it at your own CSV (a header row `x0,x1,...`, then one numeric row per
point — any extra trailing columns are ignored), or generate a reproducible set:

```sh
python tools/datasets.py --name moons  --n 1500 --out data/moons.csv
python tools/datasets.py --name varied --n 1500 --out data/varied.csv   # mixed shapes + density + noise
```

## 2. Cluster

```sh
# cluster_csv in.csv [out_prefix] [min_pts] [epsilon] [threshold] [min_cluster_frac]
cluster_csv data/moons.csv data/moons 10 -1 3.0 0.02
```

- `epsilon <= 0` auto-estimates the generating distance.
- `threshold` cuts the reachability plot into clusters (smaller = tighter clusters;
  this is the DBSCAN-style flat cut, `get_cluster_indices`).
- `min_cluster_frac` folds clusters smaller than this fraction of the cloud into noise.

It writes `<out>_points.csv` (`x0,...,cluster_id`) and `<out>_reach.csv` (the
ordering + reachability), and prints the cluster/noise counts.

## 3. See it

```sh
python tools/visualize.py --points data/moons_points.csv --reach data/moons_reach.csv
```

The scatter is colored by cluster (2-D/3-D directly, higher-D via a PCA projection);
the reachability plot shows the valleys (clusters) and peaks (separations). On
`moons` OPTICS recovers the two interleaving crescents — a shape k-means cannot
separate; on `varied` it isolates the dense and elongated clusters and leaves the
sparse background as noise. See the top-level README for how to read these plots and
choose `min_pts` / `epsilon` / `threshold`.
