#!/usr/bin/env python3
"""Reproducible 2-D synthetic datasets for the OPTICS examples, the k-means/DBSCAN
comparison, and the scikit-learn cross-validation.

Pure-numpy, deterministic-by-seed generators (no scikit-learn needed to *produce*
the data, so figures and numbers are regenerable anywhere). Each generator emits
the same CSV contract the C++ examples read and ``tools/visualize.py`` renders:

  * a coordinates file  -- header ``x0,x1`` then one ``x,y`` row per point
  * a ground-truth file -- header ``label`` then one int per point (-1 = noise)

The coordinates file is the clustering *input*; the truth file is used to color
"true" clusters and to score our labels against scikit-learn (see
``tools/validate_sklearn.py``).

Usage:
  python tools/datasets.py --name moons  --n 1000 --out data/moons.csv
  python tools/datasets.py --name blobs  --n 900  --out data/blobs.csv
  python tools/datasets.py --name varied --n 1500 --out data/varied.csv

The truth file path defaults to ``<out stem>_truth.csv`` next to ``--out``.
"""

import argparse
import csv
import math
import os

import numpy as np


def make_blobs(n=900, centers=None, std=0.6, seed=0):
    """Isotropic Gaussian blobs of equal population."""
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = [(-6, -6), (0, 5), (6, -1), (7, 6)]
    centers = np.asarray(centers, dtype=float)
    k = len(centers)
    per = n // k
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(rng.normal(c, std, size=(per, 2)))
        ys += [i] * per
    return np.vstack(xs), np.asarray(ys, dtype=int)


def make_moons(n=1000, noise=0.08, seed=0):
    """Two interleaving half circles -- the classic non-convex test where k-means
    fails and density methods succeed."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0.0, math.pi, n1)
    t2 = np.linspace(0.0, math.pi, n2)
    outer = np.c_[np.cos(t1), np.sin(t1)]
    inner = np.c_[1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5]
    x = np.vstack([outer, inner]).astype(float)
    x += rng.normal(0.0, noise, x.shape)
    y = np.asarray([0] * n1 + [1] * n2, dtype=int)
    return x * 10.0, y  # scale up so the auto-epsilon lands in a sensible range


def make_varied(n=1500, seed=0):
    """Chameleon-inspired: clusters of different shape and density plus uniform
    noise -- exactly the setting OPTICS handles and a single global k (or eps)
    does not. (Not the original CHAMELEON files, which are not bundled.)"""
    rng = np.random.default_rng(seed)
    parts, labels = [], []

    # 0: dense compact blob
    n0 = n // 4
    parts.append(rng.normal((-10, -8), 0.5, size=(n0, 2)))
    labels += [0] * n0

    # 1: sparse, wide blob
    n1 = n // 4
    parts.append(rng.normal((10, 8), 2.2, size=(n1, 2)))
    labels += [1] * n1

    # 2: elongated (anisotropic) blob
    n2 = n // 4
    blob = rng.normal((0, 0), 1.0, size=(n2, 2))
    blob = blob @ np.array([[4.0, 0.0], [0.0, 0.4]])  # stretch along x
    blob += np.array([0, 10])
    parts.append(blob)
    labels += [2] * n2

    # noise: uniform over the bounding region
    n_noise = n - (n0 + n1 + n2)
    parts.append(rng.uniform(low=(-16, -14), high=(16, 16), size=(n_noise, 2)))
    labels += [-1] * n_noise

    x = np.vstack(parts).astype(float)
    y = np.asarray(labels, dtype=int)
    # shuffle so order does not encode the labels
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def make_density(n=1500, seed=0):
    """The canonical varying-density failure for a single global density.

    Two tight, dense clusters sit close together; a third, very sparse cluster has
    internal spacing *larger than the gap* between the dense pair. So any DBSCAN
    ``eps`` small enough to keep the dense pair apart shatters the sparse cluster into
    noise, and any ``eps`` large enough to connect the sparse cluster merges the dense
    pair — no single ``eps`` recovers all three. OPTICS reads all three out of the
    reachability hierarchy (the Xi method), and k-means (fixed k, convex) ignores the
    densities and the noise entirely."""
    rng = np.random.default_rng(seed)
    dense = max(1, int(round(n * 0.25)))
    sparse = max(1, int(round(n * 0.32)))
    noise = max(1, int(round(n * 0.05)))
    parts, labels = [], []
    # Two very tight, dense clusters with a small gap between them.
    parts.append(rng.normal((-1.0, 0.0), 0.22, size=(dense, 2)));  labels += [0] * dense
    parts.append(rng.normal(( 1.0, 0.0), 0.22, size=(dense, 2)));  labels += [1] * dense
    # A *uniform-density* disk for the sparse cluster: solid enough to be one cluster,
    # but with point spacing larger than the dense pair's gap -- so any eps big enough
    # to connect the disk also bridges the dense pair, and any eps that keeps the pair
    # apart shatters the disk. No single DBSCAN eps recovers all three.
    radius = rng.uniform(0.0, 1.0, sparse) ** 0.5 * 6.0
    angle = rng.uniform(0.0, 2.0 * math.pi, sparse)
    parts.append(np.column_stack([0.0 + radius * np.cos(angle), 9.0 + radius * np.sin(angle)]))
    labels += [2] * sparse
    parts.append(rng.uniform(low=(-7.0, -4.0), high=(7.0, 16.0), size=(noise, 2)))
    labels += [-1] * noise
    x = np.vstack(parts).astype(float)
    y = np.asarray(labels, dtype=int)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


GENERATORS = {
    "blobs": make_blobs,
    "moons": make_moons,
    "varied": make_varied,
    "density": make_density,
}


def write_csv(coords, labels, coords_path, truth_path=None):
    """Write the coordinates file (header x0,x1) and the truth file (header label)."""
    os.makedirs(os.path.dirname(os.path.abspath(coords_path)), exist_ok=True)
    dim = coords.shape[1]
    with open(coords_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"x{i}" for i in range(dim)])
        for row in coords:
            w.writerow([f"{v:.6g}" for v in row])
    if truth_path is None:
        stem, ext = os.path.splitext(coords_path)
        truth_path = f"{stem}_truth{ext or '.csv'}"
    with open(truth_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"])
        for v in labels:
            w.writerow([int(v)])
    return coords_path, truth_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate reproducible 2-D datasets (CSV).")
    p.add_argument("--name", choices=sorted(GENERATORS), required=True)
    p.add_argument("--n", type=int, default=1000, help="approximate number of points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="coordinates CSV path (default data/<name>.csv)")
    args = p.parse_args(argv)

    out = args.out or os.path.join("data", f"{args.name}.csv")
    coords, labels = GENERATORS[args.name](n=args.n, seed=args.seed)
    coords_path, truth_path = write_csv(coords, labels, out)
    n_clusters = len(set(int(v) for v in labels) - {-1})
    n_noise = int((labels < 0).sum())
    print(f"wrote {coords_path} ({len(coords)} points, {coords.shape[1]}-D)")
    print(f"wrote {truth_path} ({n_clusters} true clusters, {n_noise} noise points)")


if __name__ == "__main__":
    main()
