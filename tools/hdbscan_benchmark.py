#!/usr/bin/env python3
"""HDBSCAN* cross-check harness: score this library's exact ``hdbscan`` and approximate
``shdbscan`` against scikit-learn's ``HDBSCAN`` and ground-truth labels (ARI / NMI / Rand),
plus a direct **label-agreement** column (ours-HDBSCAN vs sk-HDBSCAN) -- the actual
correctness check for the reimplementation (issue #52).

This is the HDBSCAN counterpart to ``tools/quality_benchmark.py`` (which covers OPTICS /
sOPTICS). Our side runs through the ``hdbscan_compare`` C++ harness (build it first); the
scikit-learn reference and the metrics come from scikit-learn.

CONVENTION: ``min_samples`` is SELF-INCLUSIVE in both this library and
``sklearn.cluster.HDBSCAN`` (the point is its own 1st neighbour), so the harness passes the
SAME (min_cluster_size, min_samples) to both -- an apples-to-apples comparison. (The older
scikit-learn-contrib/hdbscan package excludes self, so it would need min_samples+1.)

``shdbscan`` is a COSINE method (points are L2-normalized internally), so on the raw-Euclidean
2-D toys it is expected to score lower than exact ``hdbscan``; it is in its element on the
cos-blobs-* rows. That is the metric, not a defect.

Build:  cmake --build --preset msvc --target hdbscan_compare
Usage:  python tools/hdbscan_benchmark.py --exe build/test/Release/hdbscan_compare
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # quiet sklearn deprecation chatter

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as ds  # tools/datasets.py


def make_cosine_blobs(n=1200, dim=8, k=5, jitter=0.12, seed=0):
    """k clusters along distinct random directions, L2-normalized onto the unit sphere -- so
    cluster identity is angular and the cosine method (shdbscan) is in the right metric."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    per = n // k
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(c + rng.normal(0.0, jitter, size=(per, dim)))
        ys += [i] * per
    x = np.vstack(xs).astype(float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    perm = rng.permutation(len(x))
    return x[perm], np.asarray(ys, dtype=int)[perm]


def datasets_list(seed=0):
    """(name, X, truth) tuples: Euclidean 2-D toys + cosine high-D clouds."""
    out = []
    for name in ("blobs", "moons", "varied", "density"):
        X, y = ds.GENERATORS[name](n=1200, seed=seed)
        out.append((f"{name}-2d", X, y))
    out.append(("cos-blobs-8d", *make_cosine_blobs(n=1200, dim=8, k=5, seed=seed)))
    out.append(("cos-blobs-16d", *make_cosine_blobs(n=1200, dim=16, k=6, seed=seed)))
    return out


def load_franti(franti_dir):
    """Franti shape sets fetched by tools/fetch_datasets.py -> (name, X, truth) tuples."""
    out = []
    if not os.path.isdir(franti_dir):
        return out
    for cpath in sorted(glob.glob(os.path.join(franti_dir, "*_coords.csv"))):
        name = os.path.basename(cpath)[: -len("_coords.csv")]
        tpath = os.path.join(franti_dir, name + "_truth.csv")
        if not os.path.isfile(tpath):
            continue
        X = np.loadtxt(cpath, delimiter=",", skiprows=1, ndmin=2)
        truth = np.loadtxt(tpath, delimiter=",", skiprows=1, dtype=int, ndmin=1)
        out.append((f"franti:{name}", X, truth))
    return out


def write_coords(X, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"x{i}" for i in range(X.shape[1])])
        for row in X:
            w.writerow([f"{v:.6g}" for v in row])


def run_ours(exe, path, min_cluster_size, min_samples):
    """Returns (hdbscan_labels, shdbscan_labels, hdbscan_ms, shdbscan_ms)."""
    r = subprocess.run([exe, path, str(min_cluster_size), str(min_samples)],
                       capture_output=True, text=True, check=True)
    h, s = [], []
    for row in csv.DictReader(r.stdout.splitlines()):
        h.append(int(row["hdbscan"]))
        s.append(int(row["shdbscan"]))
    h_ms = s_ms = float("nan")
    for tok in r.stderr.split():
        if tok.startswith("hdbscan_ms="):
            h_ms = float(tok.split("=")[1])
        elif tok.startswith("shdbscan_ms="):
            s_ms = float(tok.split("=")[1])
    return np.asarray(h), np.asarray(s), h_ms, s_ms


def run_sklearn_hdbscan(X, min_cluster_size, min_samples):
    """scikit-learn HDBSCAN at the SAME (min_cluster_size, min_samples); both self-inclusive."""
    t = time.perf_counter()
    m = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(X)
    return m.labels_, (time.perf_counter() - t) * 1000.0


def scores(truth, pred):
    return (adjusted_rand_score(truth, pred),
            normalized_mutual_info_score(truth, pred),
            rand_score(truth, pred))


def main(argv=None):
    p = argparse.ArgumentParser(description="HDBSCAN* cross-check vs scikit-learn + ground truth.")
    p.add_argument("--exe", required=True, help="path to hdbscan_compare")
    p.add_argument("--min-cluster-size", type=int, default=15)
    p.add_argument("--min-samples", type=int, default=0, help="0 => use min_cluster_size (sklearn default)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--franti-dir", default=os.path.join("data", "franti"),
                   help="dir with Franti *_coords.csv/*_truth.csv (run tools/fetch_datasets.py)")
    args = p.parse_args(argv)

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"harness not found: {exe} (build target hdbscan_compare)", file=sys.stderr)
        return 2

    mcs = args.min_cluster_size
    ms = args.min_samples if args.min_samples > 0 else mcs  # mirror the C++ default

    methods = ["ours-HDBSCAN", "ours-sHDBSCAN", "sk-HDBSCAN"]
    data = datasets_list() + load_franti(args.franti_dir)
    names = [d[0] for d in data]
    ari, nmi, rand, agree, timing = {}, {}, {}, {}, {}

    for name, X, truth in data:
        path = os.path.join(args.data_dir, f"hb_{name.replace(':', '_')}.csv")
        write_coords(X, path)
        h_lab, s_lab, h_ms, s_ms = run_ours(exe, path, mcs, ms)
        sk_lab, sk_ms = run_sklearn_hdbscan(X, mcs, ms)
        preds = {"ours-HDBSCAN": h_lab, "ours-sHDBSCAN": s_lab, "sk-HDBSCAN": sk_lab}

        for m, lab in preds.items():
            a, nm, rd = scores(truth, lab)
            ari[(name, m)], nmi[(name, m)], rand[(name, m)] = a, nm, rd
        # The cross-check: how closely our exact HDBSCAN reproduces scikit-learn's labelling.
        agree[name] = adjusted_rand_score(sk_lab, h_lab)
        timing[name] = {"ours-HDBSCAN": h_ms, "ours-sHDBSCAN": s_ms, "sk-HDBSCAN": sk_ms}

    def table(title, store):
        print(f"\n=== {title} (vs ground truth; 1.0 = perfect) ===")
        head = f"{'dataset':18s} " + " ".join(f"{m:>14s}" for m in methods)
        print(head)
        print("-" * len(head))
        for nm in names:
            cells = " ".join(f"{store[(nm, m)]:14.3f}" for m in methods)
            print(f"{nm:18s} {cells}")

    table("Adjusted Rand Index", ari)
    table("Normalized Mutual Information", nmi)
    table("Rand Index", rand)

    print("\n=== cross-check: ours-HDBSCAN vs sk-HDBSCAN (ARI; 1.0 = identical labelling) ===")
    for nm in names:
        print(f"{nm:18s} {agree[nm]:14.3f}")

    print("\n=== time (ms; lower is better) ===")
    head = f"{'dataset':18s} " + " ".join(f"{m:>14s}" for m in methods)
    print(head)
    print("-" * len(head))
    for nm in names:
        cells = " ".join(f"{timing[nm].get(m, float('nan')):14.0f}" for m in methods)
        print(f"{nm:18s} {cells}")

    print(f"\nParams: min_cluster_size={mcs}, min_samples={ms} (both self-inclusive, passed identically")
    print("to ours and scikit-learn). The cross-check column is the correctness signal -- our exact")
    print("hdbscan should track sk-HDBSCAN closely (tie-breaking/condensing details differ, so compare")
    print("clusters, not bit-identical labels). shdbscan is COSINE: strong on cos-blobs-*, weaker on the")
    print("Euclidean 2-D toys. Franti sets: cs.uef.fi/sipu/datasets (fetch with tools/fetch_datasets.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
