#!/usr/bin/env python3
"""Clustering-quality harness: score this library (OPTICS + sOPTICS) and scikit-learn
(OPTICS + HDBSCAN) against ground-truth labels with ARI / NMI / Rand.

This is the quality counterpart to ``tools/timing_compare.py``. It also prints a small
timing table (our OPTICS / sOPTICS, from the C++ harness, plus scikit-learn OPTICS), the
runnable part of the CPU-comparison tracker (#53). External engines (ELKI, mhahsler/dbscan,
NinhPham/sDbscan) are *not* invoked here -- see ``tools/README.md`` for how to add them.

Datasets:
  * Euclidean 2-D toys (blobs, moons, varied, density) from ``datasets.py`` -- where OPTICS
    and HDBSCAN belong; sOPTICS (a COSINE method) is reported too, but is expected to score
    lower here (its labels are direction-based, so a Euclidean layout is the wrong metric).
  * cosine-appropriate high-D clouds (clusters along distinct directions, L2-normalized) --
    sOPTICS's intended regime, where it should match the exact methods.

Our side runs through the ``optics_quality_compare`` C++ harness (build it first); ARI/NMI/
Rand come from scikit-learn. Build:  cmake --build --preset msvc --target optics_quality_compare

Usage:
  python tools/quality_benchmark.py --exe build/test/Release/optics_quality_compare
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # quiet sklearn deprecation chatter

import numpy as np
from sklearn.cluster import OPTICS, HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as ds  # tools/datasets.py


def make_cosine_blobs(n=1200, dim=8, k=5, jitter=0.12, seed=0):
    """k clusters along distinct random directions in R^dim, L2-normalized onto the unit
    sphere -- so cluster identity is angular and cosine (sOPTICS) is the right metric."""
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
    """(name, X, truth) tuples."""
    out = []
    for name in ("blobs", "moons", "varied", "density"):
        X, y = ds.GENERATORS[name](n=1200, seed=seed)
        out.append((f"{name}-2d", X, y))
    out.append(("cos-blobs-8d", *make_cosine_blobs(n=1200, dim=8, k=5, seed=seed)))
    out.append(("cos-blobs-16d", *make_cosine_blobs(n=1200, dim=16, k=6, seed=seed)))
    return out


def write_coords(X, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"x{i}" for i in range(X.shape[1])])
        for row in X:
            w.writerow([f"{v:.6g}" for v in row])


def run_ours(exe, path, min_pts, chi):
    """Returns (optics_labels, soptics_labels, optics_ms, soptics_ms, eps)."""
    r = subprocess.run([exe, path, str(min_pts), str(chi)], capture_output=True, text=True, check=True)
    opt, sopt = [], []
    reader = csv.DictReader(r.stdout.splitlines())
    for row in reader:
        opt.append(int(row["optics"]))
        sopt.append(int(row["soptics"]))
    o_ms = s_ms = eps = float("nan")
    for tok in r.stderr.split():
        if tok.startswith("optics_ms="):
            o_ms = float(tok.split("=")[1])
        elif tok.startswith("soptics_ms="):
            s_ms = float(tok.split("=")[1])
        elif tok.startswith("eps="):
            eps = float(tok.split("=")[1])
    return np.asarray(opt), np.asarray(sopt), o_ms, s_ms, eps


def run_sklearn_optics(X, min_pts, chi):
    t = time.perf_counter()
    m = OPTICS(min_samples=min_pts, cluster_method="xi", xi=chi).fit(X)
    return m.labels_, (time.perf_counter() - t) * 1000.0


def run_sklearn_hdbscan(X, min_pts):
    m = HDBSCAN(min_cluster_size=max(5, min_pts)).fit(X)
    return m.labels_


def scores(truth, pred):
    return (adjusted_rand_score(truth, pred),
            normalized_mutual_info_score(truth, pred),
            rand_score(truth, pred))


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


def find_rscript(arg):
    if arg:
        return arg if os.path.isfile(arg) else None
    p = shutil.which("Rscript")
    if p:
        return p
    base = r"C:\Program Files\R"
    if os.path.isdir(base):
        for d in sorted(os.listdir(base), reverse=True):  # newest R first
            cand = os.path.join(base, d, "bin", "Rscript.exe")
            if os.path.isfile(cand):
                return cand
    return None


def run_dbscan_r(rscript, script, coords_path, min_pts, xi, eps):
    """Run mhahsler/dbscan OPTICS+Xi via Rscript at the given eps; returns (labels, ms)."""
    out_csv = os.path.splitext(coords_path)[0] + "_dbscanR.csv"
    eps_arg = str(eps) if eps and eps == eps and eps > 0 else "-1"  # eps==eps filters NaN
    r = subprocess.run([rscript, script, coords_path, out_csv, str(min_pts), str(xi), eps_arg],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out_csv):
        sys.stderr.write(r.stderr)
        return None, float("nan")
    labels = []
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
    ms = float("nan")
    for tok in r.stderr.split():
        if tok.startswith("dbscan_ms="):
            ms = float(tok.split("=")[1])
    return np.asarray(labels), ms


def dbscan_r_available(rscript):
    """True iff Rscript can actually load the dbscan package (not just exist on disk)."""
    r = subprocess.run([rscript, "-e", "suppressMessages(library(dbscan))"], capture_output=True, text=True)
    return r.returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser(description="ARI/NMI/Rand quality benchmark vs ground truth.")
    p.add_argument("--exe", required=True, help="path to optics_quality_compare")
    p.add_argument("--min-pts", type=int, default=10)
    p.add_argument("--chi", type=float, default=0.05)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--franti-dir", default=os.path.join("data", "franti"),
                   help="dir with Franti *_coords.csv/*_truth.csv (run tools/fetch_datasets.py)")
    p.add_argument("--rscript", default=None, help="path to Rscript (auto-detected if omitted)")
    p.add_argument("--no-dbscan-r", action="store_true", help="skip the mhahsler/dbscan (R) column")
    args = p.parse_args(argv)

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"harness not found: {exe} (build target optics_quality_compare)", file=sys.stderr)
        return 2

    rscript = None if args.no_dbscan_r else find_rscript(args.rscript)
    r_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_dbscan_r.R")
    use_r = bool(rscript) and os.path.isfile(r_script_path) and dbscan_r_available(rscript)
    if rscript and not use_r:
        print("dbscan-R: Rscript found but the 'dbscan' package is not loadable -- skipping its column.\n"
              "          install it with:  Rscript -e \"install.packages('dbscan')\"\n"
              "          (if R is in a sandbox, set R_LIBS_USER to the library holding dbscan)", file=sys.stderr)
    print(f"dbscan-R: {'using ' + rscript if use_r else 'skipped'}", file=sys.stderr)

    methods = ["ours-OPTICS", "ours-sOPTICS", "sk-OPTICS", "sk-HDBSCAN"] + (["dbscan-R"] if use_r else [])
    data = datasets_list() + load_franti(args.franti_dir)
    names = [d[0] for d in data]
    ari, nmi, rand, timing = {}, {}, {}, {}

    for name, X, truth in data:
        path = os.path.join(args.data_dir, f"qb_{name.replace(':', '_')}.csv")
        write_coords(X, path)
        o_lab, s_lab, o_ms, s_ms, eps = run_ours(exe, path, args.min_pts, args.chi)
        sk_lab, sk_ms = run_sklearn_optics(X, args.min_pts, args.chi)
        hd_lab = run_sklearn_hdbscan(X, args.min_pts)
        preds = {"ours-OPTICS": o_lab, "ours-sOPTICS": s_lab, "sk-OPTICS": sk_lab, "sk-HDBSCAN": hd_lab}
        tcols = {"ours-OPTICS": o_ms, "ours-sOPTICS": s_ms, "sk-OPTICS": sk_ms}

        if use_r:
            # same eps as ours-OPTICS, so the dbscan-R timing is comparable (same work).
            r_lab, r_ms = run_dbscan_r(rscript, r_script_path, path, args.min_pts, args.chi, eps)
            if r_lab is not None and len(r_lab) == len(truth):
                preds["dbscan-R"], tcols["dbscan-R"] = r_lab, r_ms
            else:
                preds["dbscan-R"], tcols["dbscan-R"] = np.full(len(truth), -1), float("nan")

        for m, lab in preds.items():
            a, nm, rd = scores(truth, lab)
            ari[(name, m)], nmi[(name, m)], rand[(name, m)] = a, nm, rd
        timing[name] = tcols

    def table(title, store):
        print(f"\n=== {title} (vs ground truth; 1.0 = perfect) ===")
        head = f"{'dataset':18s} " + " ".join(f"{m:>13s}" for m in methods)
        print(head)
        print("-" * len(head))
        for nm in names:
            cells = " ".join(f"{store[(nm, m)]:13.3f}" for m in methods)
            print(f"{nm:18s} {cells}")

    table("Adjusted Rand Index", ari)
    table("Normalized Mutual Information", nmi)
    table("Rand Index", rand)

    print("\n=== ordering time (ms; lower is better) ===")
    tmeth = ["ours-OPTICS", "ours-sOPTICS", "sk-OPTICS"] + (["dbscan-R"] if use_r else [])
    head = f"{'dataset':18s} " + " ".join(f"{m:>13s}" for m in tmeth)
    print(head)
    print("-" * len(head))
    for nm in names:
        cells = " ".join(f"{timing[nm].get(m, float('nan')):13.0f}" for m in tmeth)
        print(f"{nm:18s} {cells}")

    print("\nNotes: sOPTICS is a COSINE method (strong on cos-blobs-*, lower on Euclidean layouts).")
    print("ours-OPTICS and dbscan-R are both exact-Euclidean OPTICS run at the SAME eps (fair")
    print("timing); tie-breaking differs, so compare clusters (ARI/NMI/Rand), not bit-identical")
    print("orderings. Franti datasets: cs.uef.fi/sipu/datasets (fetch with tools/fetch_datasets.py).")
    return 0

    print("\nNote: sOPTICS is a COSINE method (direction-based). It is expected to score well on the\n"
          "cos-blobs-* rows and lower on the Euclidean 2-D toys -- that is the metric, not a defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
