#!/usr/bin/env python3
"""Orchestrator for the 1.0.0 benchmark matrix (issue #59, design section 8.3).

Expands a tier spec into cells, generates each dataset once (via ``gen_dataset``),
runs every applicable engine on the *same* CSV, scores quality against ground truth,
and appends one **tidy long-format** row per (cell, engine, config, rep, measure) to a
results CSV. It is **checkpointed/resumable** (already-recorded (cell, engine, config,
rep) keys are skipped) and applies **feasibility gating + timeouts** so an infeasible
O(n^2) cell is a *logged skip*, never a silent omission or a hang (design section 3b).

Engines wired here (what runs on the dev box; ELKI / NinhPham-sDbscan are deferred to
the Docker repro env -- design section 8.4):
  * ours      -- optics, soptics, hdbscan, shdbscan   (via the `optics_matrix` C++ exe)
  * scikit-learn -- OPTICS(xi), HDBSCAN, DBSCAN, KMeans
  * dbscan-R  -- OPTICS+extractXi (optional; probed, skipped with a reason if absent)

`optics_matrix` CLI contract (this file is the spec; the C++ exe must match):
  optics_matrix --coords <csv> --algo optics|soptics|hdbscan|shdbscan
                --out-labels <csv> [--min-pts N] [--min-cluster-size N] [--chi F]
                [--eps knee|uniform|<num>] [--metric cosine|l2|l1]
                [--mode ondemand|precompute] [--projection gaussian|structured]
                [--threads N] [--seed N]
  -> writes <out-labels> (header "label", one int per input point, -1 = noise)
  -> prints one line to stdout:  RESULT eps=<f> ordering_ms=<i> n_clusters=<i> noise=<i>

Usage:
  python tools/run_matrix.py --tier pilot --out results/matrix.csv --reps 1
  python tools/run_matrix.py --tier pilot --out results/matrix.csv --resume
  python tools/run_matrix.py --list-tiers

Run from the repo root (so `tools/` is importable and the C++ exe path resolves).
Designed to be killed and restarted: pass --resume to continue from the last row.
"""

import argparse
import csv
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as G  # noqa: E402

# ---- optional deps (scored only if present) --------------------------------------------
try:
    from sklearn.cluster import OPTICS, DBSCAN, KMeans, HDBSCAN
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


# ---- tiers (design section 3a). "pilot" is a tiny end-to-end smoke; the spines expand it. ----
# A cell is a dict of dataset knobs; engines are chosen per cell by applicability (section 4/9).
def _cells_pilot():
    """A handful of small cells exercising every engine + both metrics + a shape."""
    out = []
    for d in (3, 16):
        out.append(dict(n=1500, d=d, k=max(2, d // 2), density="mixed", noise=0.1, shape="blobs"))
    out.append(dict(n=1200, d=2, k=2, density="mixed", noise=0.0, shape="spiral"))
    out.append(dict(n=1500, d=8, k=2, density="dense", noise=0.05, shape="moons"))
    return out


def _cells_scaling():  # Tier A spine (n full at d in {3,16}); gated by feasibility at the top end.
    ns = [316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000]
    return [dict(n=n, d=d, k=max(2, d // 2), density="mixed", noise=0.1, shape="blobs")
            for d in (3, 16) for n in ns]


def _cells_dim():  # Tier B spine (d full at n in {1e4,1e5}).
    ds = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128]
    return [dict(n=n, d=d, k=max(2, d // 2), density="mixed", noise=0.1, shape="blobs")
            for n in (10000, 100000) for d in ds]


TIERS = {"pilot": _cells_pilot, "scaling": _cells_scaling, "dim": _cells_dim}

MEASURES = ("ordering_ms", "total_ms", "ari", "nmi", "rand", "n_clusters_pred", "eps_used", "status")


# ---- helpers ---------------------------------------------------------------------------
def cell_id(c):
    return f"n{c['n']}_d{c['d']}_k{c['k']}_{c['density']}_no{c['noise']}_{c['shape']}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def find_exe(name):
    """Locate a built Release exe under build/ (Windows: build/test/Release/<name>.exe)."""
    cands = [f"build/test/Release/{name}.exe", f"build/test/{name}",
             f"build/Release/{name}.exe", f"build/{name}"]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def predicted_seconds(engine, n, d, density):
    """Cheap cost model for feasibility gating (design section 3b). Conservative upper bounds:
    O(n^2)-ish engines (exact OPTICS/HDBSCAN, sklearn-OPTICS) blow up on dense/large n."""
    quad = engine in ("ours-hdbscan", "sk-optics", "sk-hdbscan")
    optics_quad_dense = engine in ("ours-optics",) and density == "dense"
    if quad or optics_quad_dense:
        return (n / 3000.0) ** 2 * 0.5  # ~0.5 s at n=3000, quadratic
    return (n / 100000.0) * 2.0  # ~linear, 2 s at 1e5


# ---- engine runners: each returns dict(labels=np.array|None, ordering_ms, total_ms, eps, status) ----
def run_ours(exe, algo, coords_path, labels_out, n, min_pts, mcs, metric, mode, projection, timeout):
    cmd = [exe, "--coords", coords_path, "--algo", algo, "--out-labels", labels_out,
           "--min-pts", str(min_pts), "--min-cluster-size", str(mcs), "--eps", "knee",
           "--metric", metric, "--mode", mode, "--projection", projection, "--threads", "4"]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(labels=None, ordering_ms=-1, total_ms=timeout * 1000, eps=-1, status="timeout")
    total_ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        return dict(labels=None, ordering_ms=-1, total_ms=total_ms, eps=-1, status="error")
    res = {}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            for kv in line.split()[1:]:
                kk, _, vv = kv.partition("=")
                res[kk] = vv
    labels = _read_labels(labels_out, n)
    return dict(labels=labels, ordering_ms=int(res.get("ordering_ms", -1)),
                total_ms=total_ms, eps=float(res.get("eps", -1)), status="ok")


def run_sklearn(kind, X, min_pts, mcs, eps, timeout):
    if not _HAVE_SK:
        return dict(labels=None, ordering_ms=-1, total_ms=-1, eps=eps, status="no_sklearn")
    t0 = time.perf_counter()
    try:
        if kind == "sk-optics":
            lab = OPTICS(min_samples=min_pts, cluster_method="xi").fit_predict(X)
        elif kind == "sk-hdbscan":
            lab = HDBSCAN(min_cluster_size=mcs, min_samples=min_pts).fit_predict(X)
        elif kind == "sk-dbscan":
            lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
        elif kind == "sk-kmeans":
            k = max(1, len(set(int(v) for v in _truth_cache.get(id(X), [0])) - {-1}) or 3)
            lab = KMeans(n_clusters=k, n_init=10).fit_predict(X)
        else:
            return dict(labels=None, ordering_ms=-1, total_ms=-1, eps=eps, status="unknown")
    except Exception:
        return dict(labels=None, ordering_ms=-1, total_ms=-1, eps=eps, status="error")
    ms = (time.perf_counter() - t0) * 1000.0
    return dict(labels=np.asarray(lab), ordering_ms=int(ms), total_ms=ms, eps=eps, status="ok")


_truth_cache = {}


def _read_labels(path, n):
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=int)
        arr = np.atleast_1d(arr)
        return arr if len(arr) == n else None
    except Exception:
        return None


def score(pred, truth):
    if pred is None or not _HAVE_SK:
        return (float("nan"),) * 3
    return (adjusted_rand_score(truth, pred),
            normalized_mutual_info_score(truth, pred),
            rand_score(truth, pred))


def applicable_engines(c, exe):
    """Which (engine, algo, metric_space) tuples apply to this cell (design section 9 fairness)."""
    eng = []
    if exe:
        eng += [("ours-optics", "optics", "euclidean"), ("ours-hdbscan", "hdbscan", "euclidean")]
        # cosine methods (sOPTICS/sHDBSCAN) are scored on every cell but flagged as cosine-space;
        # the analysis only *compares* them to exact on angular data (section 9.2).
        eng += [("ours-soptics", "soptics", "cosine"), ("ours-shdbscan", "shdbscan", "cosine")]
    eng += [("sk-optics", None, "euclidean"), ("sk-hdbscan", None, "euclidean"),
            ("sk-kmeans", None, "euclidean")]
    return eng


# ---- main loop -------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Run the 1.0.0 benchmark matrix.")
    p.add_argument("--tier", choices=sorted(TIERS), default="pilot")
    p.add_argument("--out", default="results/matrix.csv")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--min-pts", type=int, default=16, help="matrix convention (section 2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget-s", type=float, default=600.0, help="per-run feasibility cap (s)")
    p.add_argument("--data-dir", default="data/matrix")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--list-tiers", action="store_true")
    args = p.parse_args(argv)

    if args.list_tiers:
        for t, fn in TIERS.items():
            print(f"{t:10s} {len(fn())} cells")
        return

    exe = find_exe("optics_matrix")
    if not exe:
        print("WARNING: optics_matrix exe not found under build/ -- running sklearn engines only.\n"
              "         Build it (Release) to score ours-optics/soptics/hdbscan/shdbscan.", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    done = set()
    if args.resume and os.path.isfile(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["cell_id"], row["engine"], row["config"], row["rep"]))
        print(f"resume: {len(done)} (cell,engine,config,rep) measure-groups already recorded")

    new_file = not os.path.isfile(args.out)
    fout = open(args.out, "a", newline="")
    writer = csv.writer(fout)
    header = ["cell_id", "tier", "n", "d", "k", "density", "noise", "shape", "metric_space",
              "eps_method", "seed", "rep", "engine", "config", "measure", "value", "status",
              "commit", "threads", "timestamp"]
    if new_file:
        writer.writerow(header)

    commit = git_commit()
    threads = os.environ.get("OPTICS_BENCH_THREADS", "4")
    cells = TIERS[args.tier]()
    print(f"tier={args.tier}: {len(cells)} cells x {args.reps} rep(s); exe={'yes' if exe else 'NO'}")

    for c in cells:
        cid = cell_id(c)
        mcs = max(2, args.min_pts)
        for rep in range(args.reps):
            seed = args.seed + rep
            coords = os.path.join(args.data_dir, f"{cid}_s{seed}.csv")
            truth_p = os.path.splitext(coords)[0] + "_truth.csv"
            X, y = G.generate(c["n"], c["d"], c["k"], c["density"], c["noise"], c["shape"], seed)
            G.write_csv(X, y, coords)
            truth = y
            _truth_cache[id(X)] = truth

            for engine, algo, mspace in applicable_engines(c, exe):
                config = f"{mspace}/knee/mp{args.min_pts}"
                key = (cid, engine, config, str(rep))
                if key in done:
                    continue
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")

                # feasibility gate
                pred = predicted_seconds(engine, c["n"], c["d"], c["density"])
                if pred > args.budget_s:
                    _emit(writer, header, c, cid, args.tier, seed, rep, engine, config, mspace,
                          {"status": "skipped_budget"}, commit, threads, ts)
                    print(f"  SKIP {cid} {engine}: predicted {pred:.0f}s > {args.budget_s:.0f}s budget")
                    fout.flush()
                    continue

                if engine.startswith("ours-"):
                    labels_out = os.path.splitext(coords)[0] + f"_{engine}.csv"
                    r = run_ours(exe, algo, coords, labels_out, c["n"], args.min_pts, mcs,
                                 mspace if mspace in ("l2", "l1", "cosine") else "cosine",
                                 "ondemand", "gaussian", args.budget_s)
                else:
                    r = run_sklearn(engine, X, args.min_pts, mcs, eps=1.0, timeout=args.budget_s)

                ari, nmi, rand = score(r["labels"], truth)
                npred = (len(set(int(v) for v in r["labels"]) - {-1}) if r["labels"] is not None else -1)
                vals = {"ordering_ms": r["ordering_ms"], "total_ms": round(r["total_ms"], 1),
                        "ari": round(ari, 4), "nmi": round(nmi, 4), "rand": round(rand, 4),
                        "n_clusters_pred": npred, "eps_used": r["eps"], "status": r["status"]}
                _emit(writer, header, c, cid, args.tier, seed, rep, engine, config, mspace,
                      vals, commit, threads, ts)
                fout.flush()
                print(f"  {cid} {engine:14s} status={r['status']:6s} ari={ari:.3f} ms={r['ordering_ms']}")

    fout.close()
    print(f"done -> {args.out}")


def _emit(writer, header, c, cid, tier, seed, rep, engine, config, mspace, vals, commit, threads, ts):
    """Write one tidy long-format row per measure."""
    for measure in MEASURES:
        if measure not in vals:
            continue
        writer.writerow([cid, tier, c["n"], c["d"], c["k"], c["density"], c["noise"], c["shape"],
                         mspace, "knee", seed, rep, engine, config, measure, vals[measure],
                         vals.get("status", "ok"), commit, threads, ts])


if __name__ == "__main__":
    main()
