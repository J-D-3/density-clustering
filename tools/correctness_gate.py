#!/usr/bin/env python3
"""Pre-flight correctness gate for the benchmark matrix (issue #59, design section 8.6).

Before timing anything, confirm every engine actually *clusters* -- on easy, clean,
well-separated data it must recover the planted partition (ARI close to 1). A broken
config (wrong metric, mis-wired CLI, a regression) shows up here as a low ARI instead
of silently producing fast-but-wrong numbers in the matrix. Run this first; it exits
non-zero if any engine fails, so it can also gate a CI job or a `run_matrix.py` run.

Two clean datasets, each matched to the metric (design section 9.2 -- "metric matches
method"):
  * Euclidean: sparse blobs (rho=10, no noise)        -> exact OPTICS/HDBSCAN + sklearn.
  * Angular:   clusters along distinct unit directions -> the cosine methods (sOPTICS/sHDBSCAN).

Usage:
  python tools/correctness_gate.py                 # default thresholds
  python tools/correctness_gate.py --n 1200 --d 8
Run from the repo root.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as G  # noqa: E402
import run_matrix as RM  # noqa: E402  (reuse the engine runners + scorer)


def angular_blobs(n, d, k, seed):
    """k clusters along distinct unit directions, each at a random radius -- the clean
    cosine-separable case. Magnitude varies (radius in [0.5, 2]) so a correct cosine
    method, which ignores magnitude, still recovers them."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(k, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    per = n // k
    X, y = [], []
    for i in range(k):
        pts = dirs[i] + rng.normal(0.0, 0.05, size=(per, d))
        radius = rng.uniform(0.5, 2.0, size=(per, 1))
        X.append(pts * radius)
        y += [i] * per
    X = np.vstack(X)
    y = np.asarray(y, dtype=int)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def main(argv=None):
    p = argparse.ArgumentParser(description="Correctness gate for the benchmark-matrix engines.")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--min-pts", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--euclid-thresh", type=float, default=0.95)
    p.add_argument("--cosine-thresh", type=float, default=0.85)
    p.add_argument("--data-dir", default="data/gate")
    args = p.parse_args(argv)

    if not RM._HAVE_SK:
        print("correctness_gate: scikit-learn not available -> cannot score; aborting", file=sys.stderr)
        return 3
    exe = RM.find_exe("optics_matrix")
    os.makedirs(args.data_dir, exist_ok=True)
    mcs = max(2, args.min_pts)

    # clean Euclidean blobs (sparse, no noise) + clean angular blobs.
    Xe, ye = G.generate(args.n, args.d, args.k, "sparse", 0.0, "blobs", args.seed)
    eu_csv = os.path.join(args.data_dir, "euclid.csv")
    G.write_csv(Xe, ye, eu_csv)
    Xa, ya = angular_blobs(args.n, args.d, args.k, args.seed + 1)
    an_csv = os.path.join(args.data_dir, "angular.csv")
    G.write_csv(Xa, ya, an_csv)

    # (engine, algo|None, dataset, truth, metric_space, threshold)
    checks = []
    if exe:
        checks += [("ours-optics", "optics", eu_csv, Xe, ye, "euclidean", args.euclid_thresh),
                   ("ours-hdbscan", "hdbscan", eu_csv, Xe, ye, "euclidean", args.euclid_thresh),
                   ("ours-soptics", "soptics", an_csv, Xa, ya, "cosine", args.cosine_thresh),
                   ("ours-shdbscan", "shdbscan", an_csv, Xa, ya, "cosine", args.cosine_thresh)]
    else:
        print("WARNING: optics_matrix exe not found -- gating sklearn engines only.", file=sys.stderr)
    checks += [("sk-optics", None, eu_csv, Xe, ye, "euclidean", args.euclid_thresh),
               ("sk-hdbscan", None, eu_csv, Xe, ye, "euclidean", args.euclid_thresh)]

    failures = 0
    print(f"correctness gate: n={args.n} d={args.d} k={args.k} min_pts={args.min_pts}")
    for engine, algo, csv_path, X, truth, mspace, thresh in checks:
        if engine.startswith("ours-"):
            out = os.path.splitext(csv_path)[0] + f"_{engine}.csv"
            r = RM.run_ours(exe, algo, csv_path, out, len(truth), args.min_pts, mcs,
                            "cosine" if mspace == "cosine" else "l2", "ondemand", "gaussian", 120.0)
        else:
            r = RM.run_sklearn(engine, X, args.min_pts, mcs, eps=1.0, timeout=120.0)
        ari, _, _ = RM.score(r["labels"], truth)
        ok = (r["status"] == "ok") and (ari >= thresh)
        failures += (0 if ok else 1)
        print(f"  {'PASS' if ok else 'FAIL'}  {engine:14s} {mspace:9s} ARI={ari:.3f} "
              f"(>= {thresh:.2f})  status={r['status']}")

    if failures:
        print(f"\ncorrectness gate FAILED: {failures} engine(s) below threshold", file=sys.stderr)
        return 1
    print("\ncorrectness gate PASSED: all engines recover clean clusters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
