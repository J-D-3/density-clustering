#!/usr/bin/env python3
"""Visualize OPTICS-Clustering CSV output with matplotlib.

Reads the CSVs written by optics::io (points-with-labels and the reachability
ordering) and renders, dimension-agnostically:

  * 2D point clouds  -> 2D scatter, colored by cluster
  * 3D point clouds  -> 3D scatter (mplot3d), colored by cluster (e.g. color spaces)
  * d > 3            -> 2D scatter of the first two principal components (PCA)
  * the reachability plot, if a reachability CSV is given

This is a local/dev aid; it is not required to build or test the library.

Examples:
  python tools/visualize.py --points points.csv
  python tools/visualize.py --points points.csv --reach reach.csv --out plot.png
"""

import argparse
import csv
import sys


def load_points(path):
    """Return (coords: list[list[float]], labels: list[int], dim: int)."""
    coords, labels = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        dim = sum(1 for h in header if h.startswith("x"))
        for row in reader:
            if not row:
                continue
            coords.append([float(v) for v in row[:dim]])
            labels.append(int(row[dim]) if len(row) > dim else -1)
    return coords, labels, dim


def load_reachability(path):
    reach = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header: order_index,point_index,reachability
        for row in reader:
            if row:
                reach.append(float(row[2]))
    return reach


def project_if_needed(coords, dim):
    """Reduce to <=3 dims via PCA when dim > 3 (needs numpy)."""
    if dim <= 3:
        return coords, dim
    import numpy as np

    a = np.asarray(coords, dtype=float)
    a = a - a.mean(axis=0)
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    return (a @ vt[:2].T).tolist(), 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", required=True, help="points CSV (x0..xN,cluster_id)")
    parser.add_argument("--reach", help="reachability CSV (order_index,point_index,reachability)")
    parser.add_argument("--out", help="save figure to this path instead of showing it")
    args = parser.parse_args(argv)

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords, labels, dim = load_points(args.points)
    if not coords:
        print(f"No points found in {args.points}", file=sys.stderr)
        return 1
    coords, plot_dim = project_if_needed(coords, dim)

    n_panels = 2 if args.reach else 1
    fig = plt.figure(figsize=(7 * n_panels, 6))

    if plot_dim == 3:
        ax = fig.add_subplot(1, n_panels, 1, projection="3d")
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        ax.scatter(xs, ys, zs, c=labels, cmap="tab20", s=6)
        ax.set_title(f"Clusters ({dim}D)")
    else:
        ax = fig.add_subplot(1, n_panels, 1)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.scatter(xs, ys, c=labels, cmap="tab20", s=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Clusters ({dim}D{' -> PCA 2D' if dim > 3 else ''})")

    if args.reach:
        reach = load_reachability(args.reach)
        axr = fig.add_subplot(1, n_panels, 2)
        # Unreached points (-1) are drawn at the tallest bar so valleys stand out.
        top = max([r for r in reach if r >= 0], default=1.0)
        heights = [r if r >= 0 else top * 1.1 for r in reach]
        axr.bar(range(len(heights)), heights, width=1.0)
        axr.set_title("Reachability plot")
        axr.set_xlabel("cluster order")
        axr.set_ylabel("reachability distance")

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"Wrote {args.out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
