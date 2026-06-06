#!/usr/bin/env python3
"""Render an honest side-by-side comparison: this library's OPTICS vs k-means vs
DBSCAN on the same 2-D datasets.

The point is not "OPTICS wins" -- it is to show *where* each method fits. k-means
is fast and deterministic but needs k and assumes convex, similar-size clusters;
DBSCAN finds arbitrary shapes and noise but uses a single global density (one eps);
OPTICS needs neither k nor a single eps and exposes the full density hierarchy.

OPTICS labels come from *this library* (the cluster_csv example); k-means and
DBSCAN come from scikit-learn. Generates the datasets via tools/datasets.py if
missing, and writes a PNG.

Usage:
  python tools/compare_algorithms.py --exe build/examples/Release/cluster_csv \
      --out docs/img/algo_comparison.png
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.cluster import KMeans, DBSCAN  # noqa: E402

import datasets as ds  # tools/datasets.py (same directory)


# Per-dataset parameters (documented, picked to give each method a fair shot).
SPECS = [
    # name,    n,     true_k, optics_thresh, dbscan_eps, min_pts
    ("moons",  1500,  2,      3.0,           1.6,        10),
    ("varied", 1500,  3,      2.5,           1.2,        10),
]


def load_coords(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        dim = sum(1 for h in header if h.startswith("x")) or len(header)
        for row in r:
            if row:
                rows.append([float(v) for v in row[:dim]])
    return np.asarray(rows, dtype=float)


def load_labels(points_csv):
    labels = []
    with open(points_csv, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        dim = sum(1 for h in header if h.startswith("x"))
        for row in r:
            if row:
                labels.append(int(row[dim]))
    return np.asarray(labels, dtype=int)


def run_ours(exe, coords_csv, out_prefix, min_pts, thresh):
    cmd = [exe, coords_csv, out_prefix, str(min_pts), "-1", str(thresh), "0.01"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return load_labels(out_prefix + "_points.csv")


def n_clusters(labels):
    return len(set(int(v) for v in labels) - {-1})


def scatter(ax, X, labels, title):
    labs = np.asarray(labels)
    uniq = sorted(set(labs.tolist()))
    cmap = plt.get_cmap("tab10")
    for k in uniq:
        m = labs == k
        if k < 0:
            ax.scatter(X[m, 0], X[m, 1], s=4, c="lightgray", label="noise")
        else:
            ax.scatter(X[m, 0], X[m, 1], s=5, color=cmap(k % 10))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv=None):
    p = argparse.ArgumentParser(description="OPTICS vs k-means vs DBSCAN comparison figure.")
    p.add_argument("--exe", required=True, help="path to the built cluster_csv executable")
    p.add_argument("--out", default="docs/img/algo_comparison.png")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"cluster_csv not found: {exe} (build target cluster_csv first)", file=sys.stderr)
        return 2
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fig, axes = plt.subplots(len(SPECS), 3, figsize=(10, 3.4 * len(SPECS)))
    if len(SPECS) == 1:
        axes = axes[None, :]

    for row, (name, n, true_k, thresh, eps, min_pts) in enumerate(SPECS):
        coords_csv = os.path.join(args.data_dir, f"{name}.csv")
        if not os.path.isfile(coords_csv):
            X, _ = ds.GENERATORS[name](n=n, seed=0)
            ds.write_csv(X, np.zeros(len(X), dtype=int), coords_csv)
        X = load_coords(coords_csv)

        ours = run_ours(exe, coords_csv, os.path.join(args.data_dir, f"{name}_cmp"), min_pts, thresh)
        km = KMeans(n_clusters=true_k, n_init=10, random_state=0).fit_predict(X)
        db = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)

        scatter(axes[row][0], X, ours, f"OPTICS (this lib): {n_clusters(ours)} clusters")
        scatter(axes[row][1], X, km, f"k-means (k={true_k}, given)")
        scatter(axes[row][2], X, db, f"DBSCAN (eps={eps}): {n_clusters(db)} clusters")
        axes[row][0].set_ylabel(name, fontsize=11)

    fig.suptitle("OPTICS vs k-means vs DBSCAN", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
