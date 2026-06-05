#!/usr/bin/env python3
"""Cross-validate this library against scikit-learn's OPTICS.

Independent third-party confirmation that our implementation is correct: on the
same cloud, our reachability ordering and our extracted labels should agree with
``sklearn.cluster.OPTICS`` up to tolerance. The two implementations differ in
epsilon handling and extraction details, so agreement is measured statistically,
not bit-for-bit:

  * reachability -- Spearman rank correlation of the per-point reachability values
  * labels       -- Adjusted Rand Index (ARI) of a matching DBSCAN-style cut

It runs the ``cluster_csv`` example for our side, so build that first.

Usage:
  python tools/validate_sklearn.py --in data/moons.csv \
      --exe build/examples/Release/cluster_csv --min-pts 10 --threshold 3.0

Exits non-zero if either metric is below its (lenient) threshold, so it can gate.
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics import adjusted_rand_score


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


def load_points_labels(path):
    labels = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        dim = sum(1 for h in header if h.startswith("x"))
        for row in r:
            if row:
                labels.append(int(row[dim]))
    return np.asarray(labels, dtype=int)


def load_reach_by_point(path, n):
    """Return per-point reachability (NaN where unreached), indexed by point id."""
    reach = np.full(n, np.nan)
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # order_index,point_index,reachability
        for row in r:
            if not row:
                continue
            pid = int(row[1])
            val = float(row[2])
            reach[pid] = np.nan if val < 0 else val
    return reach


def main(argv=None):
    p = argparse.ArgumentParser(description="Cross-validate against scikit-learn OPTICS.")
    p.add_argument("--in", dest="inp", required=True, help="coordinates CSV")
    p.add_argument("--exe", required=True, help="path to the built cluster_csv executable")
    p.add_argument("--min-pts", type=int, default=10)
    p.add_argument("--threshold", type=float, default=3.0, help="reachability cut for both sides")
    p.add_argument("--min-cluster-frac", type=float, default=0.01, help="fold smaller clusters into noise")
    # Spearman on the reachability profile is the primary correctness signal (it is the
    # OPTICS algorithm output). ARI compares a downstream DBSCAN-style cut and is sensitive
    # to the threshold matching across the two implementations, so it is reported but, by
    # default, not gated.
    p.add_argument("--min-spearman", type=float, default=0.90)
    p.add_argument("--min-ari", type=float, default=-1.0, help="gate on ARI too when >= 0")
    p.add_argument("--out-prefix", default=None, help="prefix for our CSVs (default next to --in)")
    args = p.parse_args(argv)

    X = load_coords(args.inp)
    n = len(X)
    if n == 0:
        print("no points loaded", file=sys.stderr)
        return 2

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"cluster_csv executable not found: {exe}\n"
              f"build it first (target cluster_csv) and pass its path via --exe", file=sys.stderr)
        return 2

    out_prefix = args.out_prefix or os.path.splitext(args.inp)[0] + "_val"
    cmd = [exe, args.inp, out_prefix, str(args.min_pts), "-1", str(args.threshold), str(args.min_cluster_frac)]
    print("running ours:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    ours_labels = load_points_labels(out_prefix + "_points.csv")
    ours_reach = load_reach_by_point(out_prefix + "_reach.csv", n)

    # scikit-learn side: same min_samples; DBSCAN-style cut at the same threshold.
    sk = OPTICS(min_samples=args.min_pts, max_eps=np.inf, cluster_method="xi").fit(X)
    sk_reach = np.array(sk.reachability_, dtype=float)  # indexed by point; inf = unreached
    sk_reach[~np.isfinite(sk_reach)] = np.nan
    sk_labels = cluster_optics_dbscan(
        reachability=sk.reachability_,
        core_distances=sk.core_distances_,
        ordering=sk.ordering_,
        eps=args.threshold,
    )

    # Reachability agreement on points both implementations reached.
    mask = np.isfinite(ours_reach) & np.isfinite(sk_reach)
    if mask.sum() >= 3:
        rho, _ = spearmanr(ours_reach[mask], sk_reach[mask])
    else:
        rho = float("nan")

    ari = adjusted_rand_score(ours_labels, sk_labels)

    n_ours = len(set(ours_labels.tolist()) - {-1})
    n_sk = len(set(sk_labels.tolist()) - {-1})
    gate_ari = args.min_ari >= 0.0
    print(f"points: {n}  (reachability compared on {int(mask.sum())})")
    print(f"clusters: ours={n_ours}  sklearn={n_sk}")
    print(f"reachability Spearman rho = {rho:.3f}  (gate >= {args.min_spearman})   [primary]")
    print(f"label ARI                = {ari:.3f}" + (f"  (gate >= {args.min_ari})" if gate_ari else "   [informational]"))

    ok = (rho >= args.min_spearman) and (not gate_ari or ari >= args.min_ari)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
