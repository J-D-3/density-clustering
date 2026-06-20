#!/usr/bin/env python3
"""scikit-learn per-cell worker for the benchmark matrix (issue #59).

Runs ONE scikit-learn clusterer on a coords CSV and writes predicted labels + a RESULT line --
the same contract as the C++ `optics_matrix` exe. It exists as a separate process so the
orchestrator can impose a **hard wall-clock timeout** on it (design section 3b): a scikit-learn
`fit` cannot be interrupted in-process, so an under-predicted O(n^2) cell would otherwise hang the
whole run. As a subprocess it is simply killed on timeout and recorded as a timed-out data point.

stdout:  RESULT ordering_ms=<i> n_clusters=<i> noise=<i>
--out-labels file: header "label", one int per input point (-1 = noise).

Usage (invoked by run_matrix.py):
  python tools/sk_engine.py --coords c.csv --kind sk-optics --out-labels l.csv
      [--min-pts 16] [--min-cluster-size 16] [--eps 1.0] [--k 3]
"""

import argparse
import sys
import time

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description="scikit-learn matrix worker.")
    p.add_argument("--coords", required=True)
    p.add_argument("--kind", required=True,
                   choices=["sk-optics", "sk-hdbscan", "sk-dbscan", "sk-kmeans"])
    p.add_argument("--out-labels", required=True)
    p.add_argument("--min-pts", type=int, default=16)
    p.add_argument("--min-cluster-size", type=int, default=16)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--k", type=int, default=3, help="n_clusters for KMeans")
    args = p.parse_args(argv)

    from sklearn.cluster import OPTICS, DBSCAN, KMeans, HDBSCAN

    X = np.loadtxt(args.coords, delimiter=",", skiprows=1)
    # A 1-D coords file (d=1) loads as shape (n,); it must become (n, 1) -- n samples, 1 feature.
    # np.atleast_2d would wrongly make it (1, n) (one sample, n features), so reshape explicitly.
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Warm up sklearn's one-time lazy init (BLAS / threadpoolctl / first-call imports) on a tiny
    # array OUTSIDE the timed region, so the measured fit reflects steady state -- otherwise this
    # fresh subprocess would charge ~1-2 s of per-process warmup to the fit and inflate the timing.
    try:
        w = X[: min(len(X), 64)]
        OPTICS(min_samples=2, cluster_method="xi").fit_predict(w)
        HDBSCAN(min_cluster_size=2).fit_predict(w)
        DBSCAN(eps=args.eps, min_samples=2).fit_predict(w)
        KMeans(n_clusters=2, n_init=2).fit_predict(w)
    except Exception:
        pass

    t0 = time.perf_counter()
    if args.kind == "sk-optics":
        lab = OPTICS(min_samples=args.min_pts, cluster_method="xi").fit_predict(X)
    elif args.kind == "sk-hdbscan":
        lab = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_pts).fit_predict(X)
    elif args.kind == "sk-dbscan":
        lab = DBSCAN(eps=args.eps, min_samples=args.min_pts).fit_predict(X)
    else:  # sk-kmeans
        lab = KMeans(n_clusters=max(1, args.k), n_init=10).fit_predict(X)
    ms = int((time.perf_counter() - t0) * 1000.0)

    lab = np.asarray(lab, dtype=int)
    with open(args.out_labels, "w") as f:
        f.write("label\n")
        for v in lab:
            f.write(f"{int(v)}\n")
    n_clusters = len(set(int(v) for v in lab) - {-1})
    noise = int((lab < 0).sum())
    print(f"RESULT ordering_ms={ms} n_clusters={n_clusters} noise={noise}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
