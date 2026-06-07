#!/usr/bin/env python3
"""Timing comparison: this library's neighbor-search backends vs scikit-learn OPTICS,
across several test cases.

The internal backends (nanoflann exact, nanoflann approximate, and -- when built with
-DOPTICS_ENABLE_BOOST_RTREE=ON -- Boost R*-tree) are compile-time choices, so they are
timed by the C++ harness ``optics_backend_compare`` on a set of CSV clouds. This script
generates those clouds, runs the harness, then times ``sklearn.cluster.OPTICS`` on the
exact same files and prints one table (and optionally a bar chart).

All timings use 4 worker threads on our side (the harness default; OPTICS itself is
sequential apart from the neighbor precompute). scikit-learn OPTICS is single-process.

Usage:
  python tools/timing_compare.py --exe build/test/Release/optics_backend_compare \
      [--plot docs/img/timing_compare.png]
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.cluster import OPTICS

import datasets as ds  # tools/datasets.py


# Test cases: (name, generator-or-None, kwargs). 2-D cases come from datasets.py;
# the higher-D cases are uniform clouds (where exact NN gets expensive).
def make_cases(data_dir, seed=0):
    cases = []
    rng = np.random.default_rng(seed)

    for name, n in [("blobs", 2000), ("varied", 2000), ("density", 2000)]:
        path = os.path.join(data_dir, f"tc_{name}.csv")
        X, _ = ds.GENERATORS[name](n=n, seed=seed)
        ds.write_csv(X, np.zeros(len(X), dtype=int), path)
        cases.append((f"{name}-2d", path))

    for dim, n in [(3, 4000), (16, 3000)]:
        path = os.path.join(data_dir, f"tc_uniform{dim}d.csv")
        X = rng.uniform(0.0, 1000.0, size=(n, dim))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"x{i}" for i in range(dim)])
            for row in X:
                w.writerow([f"{v:.6g}" for v in row])
        cases.append((f"uniform-{dim}d", path))

    return cases


def time_internal(exe, paths, min_pts):
    """Run the C++ harness; return {(path, backend): ms}."""
    out = subprocess.run([exe, *paths, str(min_pts)], capture_output=True, text=True, check=True).stdout
    times = {}
    meta = {}
    reader = csv.DictReader(out.splitlines())
    for row in reader:
        times[(row["dataset"], row["backend"])] = float(row["ms"])
        meta[row["dataset"]] = (int(row["n"]), int(row["dim"]))
    return times, meta


def time_sklearn(path, min_pts):
    X = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        dim = sum(1 for h in header if h.startswith("x")) or len(header)
        for row in r:
            if row:
                X.append([float(v) for v in row[:dim]])
    X = np.asarray(X, dtype=float)
    OPTICS(min_samples=min_pts).fit(X[:200])  # warm up import/JIT
    t = time.perf_counter()
    OPTICS(min_samples=min_pts).fit(X)
    return math.ceil((time.perf_counter() - t) * 1000.0)  # round up to whole ms


def main(argv=None):
    p = argparse.ArgumentParser(description="Backend + scikit-learn OPTICS timing comparison.")
    p.add_argument("--exe", required=True, help="path to optics_backend_compare")
    p.add_argument("--min-pts", type=int, default=10)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--plot", default=None, help="optional output PNG (grouped bar chart)")
    args = p.parse_args(argv)

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"harness not found: {exe} (build target optics_backend_compare)", file=sys.stderr)
        return 2
    os.makedirs(args.data_dir, exist_ok=True)

    cases = make_cases(args.data_dir)
    paths = [path for _, path in cases]

    internal, meta = time_internal(exe, paths, args.min_pts)
    backends = sorted({b for (_, b) in internal})  # nanoflann, nf-approx, [boost-rtree]

    # Column order: internal backends, then sklearn.
    methods = backends + ["sklearn-OPTICS"]
    rows = []
    for label, path in cases:
        n, dim = meta.get(path, (0, 0))
        timings = {b: internal.get((path, b), float("nan")) for b in backends}
        timings["sklearn-OPTICS"] = time_sklearn(path, args.min_pts)
        rows.append((label, n, dim, timings))

    # Print a table (ms).
    head = f"{'case':14s} {'n':>6s} {'dim':>4s} " + " ".join(f"{m:>14s}" for m in methods)
    print(head)
    print("-" * len(head))
    for label, n, dim, timings in rows:
        cells = " ".join(f"{timings[m]:14.0f}" for m in methods)
        print(f"{label:14s} {n:6d} {dim:4d} {cells}")
    print("\n(ms; lower is better. Internal backends use 4 threads; sklearn is single-process.)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r[0] for r in rows]
        x = np.arange(len(labels))
        width = 0.8 / len(methods)
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, m in enumerate(methods):
            vals = [r[3][m] for r in rows]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=m)
        ax.set_yscale("log")
        ax.set_ylabel("ordering time (ms, log)")
        ax.set_title("OPTICS timing: internal backends vs scikit-learn")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
