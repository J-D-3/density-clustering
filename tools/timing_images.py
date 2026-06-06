#!/usr/bin/env python3
"""Real-world runtime analysis on RGB images (color clustering in 3-D).

For each image we sample a fixed number of pixels into an RGB point cloud (so every
method sees the *same* cloud, and so scikit-learn's OPTICS stays tractable), then time:

  * this library's internal backends -- nanoflann exact, nanoflann approximate, and
    (when the harness is built with -DOPTICS_ENABLE_BOOST_RTREE=ON) Boost R*-tree --
    via the C++ harness ``optics_backend_compare`` at 4 threads;
  * scikit-learn ``OPTICS``, ``DBSCAN``, and ``KMeans`` on the identical cloud.

Only ordering/fit time is measured (not cluster quality). Prints a table and an
optional grouped bar chart.

Usage:
  python tools/timing_images.py --exe build-boost/test/Release/optics_backend_compare \
      --images C:/Users/ingop/Pictures/airplane.ppm C:/Users/ingop/Pictures/fruits.ppm \
               C:/Users/ingop/Pictures/parrot.ppm \
      --n 8000 --plot docs/img/timing_images.png
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from sklearn.cluster import OPTICS, DBSCAN, KMeans

# Color-space clustering parameters (reasonable defaults; runtime is the focus).
MIN_PTS = 10
DBSCAN_EPS = 6.0     # RGB 0..255 distance
KMEANS_K = 8         # typical color-quantization k


def sample_to_csv(image_path, n, out_csv, seed=0):
    """Load an image, randomly sample n RGB pixels, write x0,x1,x2 CSV. Returns (n, dim)."""
    im = Image.open(image_path).convert("RGB")
    px = np.asarray(im, dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if len(px) > n:
        idx = rng.choice(len(px), size=n, replace=False)
        px = px[idx]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x0", "x1", "x2"])
        for r, g, b in px:
            w.writerow([f"{r:.0f}", f"{g:.0f}", f"{b:.0f}"])
    return len(px), 3


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                rows.append([float(v) for v in row[:3]])
    return np.asarray(rows, dtype=float)


def time_internal(exe, paths, min_pts):
    out = subprocess.run([exe, *paths, str(min_pts)], capture_output=True, text=True, check=True).stdout
    times = {}
    for row in csv.DictReader(out.splitlines()):
        times[(row["dataset"], row["backend"])] = float(row["ms"])
    return times


def timed(fn):
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000.0


def main(argv=None):
    p = argparse.ArgumentParser(description="Color-clustering runtime analysis on RGB images.")
    p.add_argument("--exe", required=True, help="path to optics_backend_compare")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--n", type=int, default=8000, help="pixels sampled per image")
    p.add_argument("--min-pts", type=int, default=MIN_PTS)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--plot", default=None)
    args = p.parse_args(argv)

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"harness not found: {exe} (build target optics_backend_compare)", file=sys.stderr)
        return 2
    os.makedirs(args.data_dir, exist_ok=True)

    # Sample each image to a CSV cloud (same cloud is reused by every method).
    cases = []  # (label, csv_path)
    for img in args.images:
        label = os.path.splitext(os.path.basename(img))[0]
        out_csv = os.path.join(args.data_dir, f"img_{label}.csv")
        n, _ = sample_to_csv(img, args.n, out_csv)
        cases.append((label, out_csv, n))

    paths = [c[1] for c in cases]
    internal = time_internal(exe, paths, args.min_pts)
    backends = sorted({b for (_, b) in internal})  # nanoflann, nf-approx, [boost-rtree]
    methods = backends + ["sklearn-OPTICS", "sklearn-DBSCAN", "sklearn-KMeans"]

    rows = []
    for label, path, n in cases:
        X = load_csv(path)
        t = {b: internal.get((path, b), float("nan")) for b in backends}
        # warm up, then time each scikit-learn estimator on the same cloud
        OPTICS(min_samples=args.min_pts).fit(X[:200])
        t["sklearn-OPTICS"] = timed(lambda: OPTICS(min_samples=args.min_pts).fit(X))
        t["sklearn-DBSCAN"] = timed(lambda: DBSCAN(eps=DBSCAN_EPS, min_samples=args.min_pts).fit(X))
        t["sklearn-KMeans"] = timed(lambda: KMeans(n_clusters=KMEANS_K, n_init=10, random_state=0).fit(X))
        rows.append((label, n, t))

    head = f"{'image':10s} {'n':>6s} " + " ".join(f"{m:>15s}" for m in methods)
    print(head)
    print("-" * len(head))
    for label, n, t in rows:
        print(f"{label:10s} {n:6d} " + " ".join(f"{t[m]:15.1f}" for m in methods))
    print("\n(ms; lower is better. Internal backends @ 4 threads; scikit-learn single-process.")
    print(f" Same sampled RGB cloud per image. min_pts={args.min_pts}, DBSCAN eps={DBSCAN_EPS}, KMeans k={KMEANS_K}.)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r[0] for r in rows]
        x = np.arange(len(labels))
        width = 0.8 / len(methods)
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, m in enumerate(methods):
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, [r[2][m] for r in rows], width, label=m)
        ax.set_yscale("log")
        ax.set_ylabel("ordering / fit time (ms, log)")
        ax.set_title(f"Color-clustering runtime on RGB images ({rows[0][1]} px each)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
