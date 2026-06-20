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
# The scikit-learn *clusterers* run in the sk_engine.py subprocess; here we only need its metrics
# to score every engine's labels against ground truth.
try:
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
    # Capped at 1e6: the design's 1e7 level needs a streaming generator/CSV writer (a 1e7x16 CSV is
    # ~2.5 GB) -- add 3162278, 10000000 here once that lands. The feasibility gate skips the exact
    # O(n^2) engines (sk-optics, exact HDBSCAN) well before the top of even this range.
    ns = [316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000]
    return [dict(n=n, d=d, k=max(2, d // 2), density="mixed", noise=0.1, shape="blobs")
            for d in (3, 16) for n in ns]


def _cells_dim():  # Tier B spine (d full at n in {1e4,1e5}).
    ds = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128]
    return [dict(n=n, d=d, k=max(2, d // 2), density="mixed", noise=0.1, shape="blobs")
            for n in (10000, 100000) for d in ds]


def _cells_axes():
    """Tier C -- the D1/D3/D4 sweep axes (design section 1). These decisions need axes the
    timing spines hold fixed, so each cell here carries an `axis` tag and is expanded (only for
    exact OPTICS) into config variants by `optics_jobs`:
      * axis='eps_mode' -> D4 (uniform vs knee eps) x D3 (OnDemand vs Precompute), exact backend.
        Density is the driver of both (eps estimators diverge by cluster structure; mode by
        neighborhood size), so we sweep sparse/dense/mixed at d in {3,16}.
      * axis='backend'  -> D1 (exact vs eps-approx{100,500,1000} vs HNSW) across the d-spine at a
        fixed moderate n, to locate the crossover dim D* (HNSW's measured ~64-D regime, n~2e4)."""
    cells = []
    for dens in ("sparse", "dense", "mixed"):
        for d in (3, 16):
            cells.append(dict(n=8000, d=d, k=max(2, d // 2), density=dens, noise=0.05,
                              shape="blobs", axis="eps_mode"))
    for d in (3, 8, 16, 32, 64, 128):
        cells.append(dict(n=20000, d=d, k=max(2, d // 2), density="mixed", noise=0.05,
                          shape="blobs", axis="backend"))
    return cells


TIERS = {"pilot": _cells_pilot, "scaling": _cells_scaling, "dim": _cells_dim, "axes": _cells_axes}

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
    """Cheap per-engine upper-bound cost model for feasibility gating (design section 3b).

    Deliberately conservative -- far better to over-estimate and skip+log a cell than to let an
    O(n^2) engine hang for hours. Constants are loose, calibrated from the pilot + the
    perf/README scaling tables; the value only has to land the right side of the budget. Engines
    that recompute neighborhoods (exact OPTICS on dense, exact HDBSCAN dense-Prim, scikit OPTICS)
    are ~O(n^2); the random-projection / linear methods are ~O(n)."""
    n = float(n)
    if engine == "sk-optics":                        # scikit OPTICS ~O(n^2), large constant
        return (n / 1000.0) ** 2 * 0.05              # ~0.05s @1k, ~5s @10k, ~500s @100k
    if engine in ("sk-hdbscan", "sk-dbscan"):        # sub-quadratic but super-linear
        return (n / 1000.0) ** 1.5 * 0.02
    if engine == "ours-hdbscan":                     # exact dense-Prim mutual-reach MST, O(n^2)
        return (n / 1000.0) ** 2 * 0.02              # ~0.02s @1k, ~2s @10k, ~200s @100k
    if engine == "ours-optics":                      # ~linear with bounded knee-eps; dense ~O(n^2)
        lin = (n / 1e5) * 2.0
        return lin * ((n / 3000.0) if density == "dense" else 1.0)
    if engine in ("ours-soptics", "ours-shdbscan"):  # CEOs random-projection build, ~linear-ish
        return (n / 1e5) * 4.0
    if engine == "sk-kmeans":                        # near-linear (Lloyd)
        return (n / 1e6) * 2.0
    return (n / 1e5) * 2.0


def reps_for(n, base_reps):
    """Auto-reduce repetitions as n grows (design section 3b: more reps at small n -> 1 at the
    top), so the scaling spine stays affordable while small cells keep a variance estimate."""
    if n <= 10_000:
        return base_reps
    if n <= 100_000:
        return max(1, base_reps // 2)
    return 1


# ---- engine runners: each returns dict(labels=np.array|None, ordering_ms, total_ms, eps, status) ----
def run_ours(exe, algo, coords_path, labels_out, n, min_pts, mcs, metric, mode, projection, timeout,
             eps="knee", backend="exact"):
    cmd = [exe, "--coords", coords_path, "--algo", algo, "--out-labels", labels_out,
           "--min-pts", str(min_pts), "--min-cluster-size", str(mcs), "--eps", eps,
           "--metric", metric, "--mode", mode, "--projection", projection,
           "--backend", backend, "--threads", "4"]
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


def run_sklearn(kind, coords_path, labels_out, n, k, min_pts, mcs, eps, timeout):
    """Run a scikit-learn clusterer in a SUBPROCESS (tools/sk_engine.py) with a hard wall-clock
    timeout. scikit-learn's fit can't be interrupted in-process, so isolating it as a subprocess is
    what lets an under-predicted O(n^2) cell be killed and recorded as timed-out rather than hanging
    the whole run (design section 3b)."""
    if not _HAVE_SK:
        return dict(labels=None, ordering_ms=-1, total_ms=-1, eps=eps, status="no_sklearn")
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sk_engine.py")
    cmd = [sys.executable, worker, "--coords", coords_path, "--kind", kind,
           "--out-labels", labels_out, "--min-pts", str(min_pts), "--min-cluster-size", str(mcs),
           "--eps", str(eps), "--k", str(k)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(labels=None, ordering_ms=-1, total_ms=timeout * 1000, eps=eps, status="timeout")
    total_ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        return dict(labels=None, ordering_ms=-1, total_ms=total_ms, eps=eps, status="error")
    res = {}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            for kv in line.split()[1:]:
                kk, _, vv = kv.partition("=")
                res[kk] = vv
    return dict(labels=_read_labels(labels_out, n), ordering_ms=int(res.get("ordering_ms", -1)),
                total_ms=total_ms, eps=eps, status="ok")


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


# Engine registry (design section 4). To ADD an engine/algorithm later -- a new backend, a new
# extraction, a future competitor -- append one (engine_name, algo|None, metric_space) tuple here
# (and, for an "ours-*" engine, teach optics_matrix the --algo, or add a runner branch). The rest
# of the pipeline (gating, scoring, tidy CSV, analysis) needs no change. This is the seam that lets
# the matrix be re-run/extended when the algorithmic infrastructure changes (see the reproducibility
# note in docs/ROADMAP-1.0.0-execution.md).
OURS_ENGINES = [("ours-optics", "optics", "euclidean"), ("ours-hdbscan", "hdbscan", "euclidean"),
                ("ours-soptics", "soptics", "cosine"), ("ours-shdbscan", "shdbscan", "cosine")]
SK_ENGINES = [("sk-optics", None, "euclidean"), ("sk-hdbscan", None, "euclidean"),
              ("sk-kmeans", None, "euclidean")]


def applicable_engines(c, exe, only=None):
    """Which (engine, algo, metric_space) tuples apply to this cell (design section 9 fairness).
    `only` (a set of engine names) restricts to those engines -- used by --engines to re-run a
    subset after an infrastructure change. cosine methods (sOPTICS/sHDBSCAN) are scored on every
    cell but flagged cosine-space; the analysis only *compares* them to exact on angular data."""
    eng = (OURS_ENGINES if exe else []) + SK_ENGINES
    if only:
        eng = [e for e in eng if e[0] in only]
    return eng


def optics_jobs(c, exe, only, min_pts):
    """Yield (engine, algo, mspace, config, eps, mode, backend) per cell.

    Normal cells: one baseline job per applicable engine -- the config string is left EXACTLY as the
    timing tiers used it (`{mspace}/knee/mp{min_pts}`), so resuming/refreshing a prior run still
    matches. Axes cells (`c['axis']` set, Tier C): exact OPTICS only, swept into the D1/D3/D4 config
    variants. The config encodes the variant (eps/mode/be), so each is a distinct, resumable row and
    `analyze_matrix.py` can group by it."""
    axis = c.get("axis")
    if not axis:
        for (e, a, m) in applicable_engines(c, exe, only):
            yield (e, a, m, f"{m}/knee/mp{min_pts}", "knee", "ondemand", "exact")
        return
    if not exe or (only and "ours-optics" not in only):
        return  # the axis sweeps are all exact-OPTICS variants
    if axis == "eps_mode":          # D4 (eps) x D3 (mode)
        for eps in ("knee", "uniform"):
            for mode in ("ondemand", "precompute"):
                yield ("ours-optics", "optics", "euclidean",
                       f"euclidean/eps={eps}/mode={mode}/be=exact/mp{min_pts}", eps, mode, "exact")
    elif axis == "backend":         # D1 (backend by dim)
        for be in ("exact", "approx100", "approx500", "approx1000", "hnsw"):
            yield ("ours-optics", "optics", "euclidean",
                   f"euclidean/eps=knee/mode=ondemand/be={be}/mp{min_pts}", "knee", "ondemand", be)


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
    p.add_argument("--engines", nargs="*", default=None,
                   help="restrict to these engine names (e.g. ours-optics ours-soptics) -- the way "
                        "to re-run a subset after changing that algorithm's code")
    p.add_argument("--refresh", action="store_true",
                   help="ignore existing checkpoint rows for the engines being run, so they re-run "
                        "and append fresh rows (newer commit/timestamp); analyze takes the latest")
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

    only = set(args.engines) if args.engines else None
    done = set()
    if (args.resume or args.refresh) and os.path.isfile(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["cell_id"], row["engine"], row["config"], row["rep"]))
        if args.refresh:
            # re-run (and re-append) the selected engines: drop their checkpoint entries so they
            # are not skipped. With no --engines filter, --refresh re-runs everything.
            before = len(done)
            done = {k for k in done if only and k[1] not in only}
            print(f"refresh: dropped {before - len(done)} checkpoint entries for re-run")
        print(f"resume/refresh: {len(done)} (cell,engine,config,rep) groups kept (skipped)")

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
    print(f"tier={args.tier}: {len(cells)} cells (reps auto-scale with n); "
          f"engines={'all' if not only else ','.join(sorted(only))}; exe={'yes' if exe else 'NO'}")

    for c in cells:
        cid = cell_id(c)
        mcs = max(2, args.min_pts)
        n_reps = reps_for(c["n"], args.reps)  # fewer reps as n grows (section 3b)
        for rep in range(n_reps):
            seed = args.seed + rep
            jobs = [j for j in optics_jobs(c, exe, only, args.min_pts)
                    if (cid, j[0], j[3], str(rep)) not in done]
            if not jobs:
                continue  # nothing to run for this (cell, rep) -- skip generation entirely
            coords = os.path.join(args.data_dir, f"{cid}_s{seed}.csv")
            X, y = G.generate(c["n"], c["d"], c["k"], c["density"], c["noise"], c["shape"], seed)
            G.write_csv(X, y, coords)
            truth = y
            true_k = len(set(int(v) for v in y) - {-1})  # KMeans n_clusters

            for engine, algo, mspace, config, eps, mode, backend in jobs:
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")

                # feasibility gate
                pred = predicted_seconds(engine, c["n"], c["d"], c["density"])
                if pred > args.budget_s:
                    _emit(writer, header, c, cid, args.tier, seed, rep, engine, config, mspace,
                          {"status": "skipped_budget"}, commit, threads, ts)
                    print(f"  SKIP {cid} {engine} [{config}]: predicted {pred:.0f}s > {args.budget_s:.0f}s budget")
                    fout.flush()
                    continue

                tag = config.replace("/", "_").replace("=", "-")
                labels_out = os.path.splitext(coords)[0] + f"_{engine}_{tag}.csv"
                if engine.startswith("ours-"):
                    r = run_ours(exe, algo, coords, labels_out, c["n"], args.min_pts, mcs,
                                 mspace if mspace in ("l2", "l1", "cosine") else "cosine",
                                 mode, "gaussian", args.budget_s, eps=eps, backend=backend)
                else:
                    r = run_sklearn(engine, coords, labels_out, c["n"], true_k, args.min_pts, mcs,
                                    eps=1.0, timeout=args.budget_s)

                ari, nmi, rand = score(r["labels"], truth)
                npred = (len(set(int(v) for v in r["labels"]) - {-1}) if r["labels"] is not None else -1)
                vals = {"ordering_ms": r["ordering_ms"], "total_ms": round(r["total_ms"], 1),
                        "ari": round(ari, 4), "nmi": round(nmi, 4), "rand": round(rand, 4),
                        "n_clusters_pred": npred, "eps_used": r["eps"], "status": r["status"]}
                _emit(writer, header, c, cid, args.tier, seed, rep, engine, config, mspace,
                      vals, commit, threads, ts)
                fout.flush()
                cfgnote = f" [{config}]" if c.get("axis") else ""
                print(f"  {cid} {engine:14s}{cfgnote} status={r['status']:6s} "
                      f"ari={ari:.3f} ms={r['ordering_ms']}")

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
