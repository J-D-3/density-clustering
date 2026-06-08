#!/usr/bin/env python3
"""Analysis for the 1.0.0 benchmark matrix (issue #59, design section 8.5).

Reads the tidy long-format CSV produced by ``run_matrix.py`` and emits the decision
tables the study exists to resolve (D1-D5), the headline speedup-vs-sklearn-OPTICS
table, and a per-cell quality table -- as Markdown to stdout (and optionally a file).

It is deliberately **honest about insufficient data**: a decision that needs an axis
the current run did not sweep (e.g. D1 needs multiple backends, D3 needs Precompute vs
OnDemand, D4 needs uniform vs knee) is reported as "insufficient data -- needs <axis>"
rather than guessed. As the tiers fill in, the same script resolves more of D1-D5.

Pure stdlib + numpy (no pandas dependency, so it runs wherever the matrix ran).

Usage:
  python tools/analyze_matrix.py results/matrix_pilot.csv
  python tools/analyze_matrix.py results/matrix.csv --out results/report.md
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

NUMERIC = {"ordering_ms", "total_ms", "ari", "nmi", "rand", "n_clusters_pred", "eps_used"}
CELL_KEYS = ("n", "d", "k", "density", "noise", "shape")


def load(path):
    """Return (cells, agg): cells[cell_id]=metadata; agg[cell_id][engine][measure]=median over reps."""
    rows = list(csv.DictReader(open(path, newline="")))
    cells = {}
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    status = defaultdict(dict)
    for r in rows:
        cid, eng, meas = r["cell_id"], r["engine"], r["measure"]
        cells.setdefault(cid, {k: r[k] for k in CELL_KEYS} | {"metric_space": r["metric_space"]})
        if meas == "status":
            status[cid][eng] = r["value"]
            continue
        if meas in NUMERIC:
            try:
                bucket[cid][eng][meas].append(float(r["value"]))
            except ValueError:
                pass
    agg = defaultdict(lambda: defaultdict(dict))
    for cid in bucket:
        for eng in bucket[cid]:
            for meas, vals in bucket[cid][eng].items():
                agg[cid][eng][meas] = float(np.median(vals)) if vals else float("nan")
            agg[cid][eng]["status"] = status[cid].get(eng, "ok")
    return cells, agg


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _engines(agg):
    seen = []
    for cid in agg:
        for e in agg[cid]:
            if e not in seen:
                seen.append(e)
    return seen


def quality_table(cells, agg, out):
    engs = _engines(agg)
    out.append("## Quality (ARI vs ground truth)\n")
    out.append("| cell | " + " | ".join(engs) + " |")
    out.append("|" + "---|" * (len(engs) + 1))
    for cid in sorted(agg):
        row = [cid]
        for e in engs:
            row.append(_fmt(agg[cid].get(e, {}).get("ari")))
        out.append("| " + " | ".join(row) + " |")
    out.append("")


def speedup_table(cells, agg, out):
    out.append("## Speed (ordering ms) + speedup vs scikit-learn OPTICS\n")
    out.append("| cell | ours-optics | sk-optics | speedup | ours-hdbscan | sk-hdbscan |")
    out.append("|---|---|---|---|---|---|")
    for cid in sorted(agg):
        oo = agg[cid].get("ours-optics", {}).get("ordering_ms")
        so = agg[cid].get("sk-optics", {}).get("ordering_ms")
        oh = agg[cid].get("ours-hdbscan", {}).get("ordering_ms")
        sh = agg[cid].get("sk-hdbscan", {}).get("ordering_ms")
        spd = (so / oo) if (oo and so and oo > 0) else None
        out.append(f"| {cid} | {_fmt(oo,1)} | {_fmt(so,1)} | "
                   f"{(_fmt(spd,1)+'x') if spd else 'n/a'} | {_fmt(oh,1)} | {_fmt(sh,1)} |")
    out.append("")


def crossover_table(cells, agg, out, exact, approx, decision, regime_note):
    """D2 (optics vs soptics) / D5 (hdbscan vs shdbscan): time ratio + ARI delta per cell.
    The approximate (cosine) method is only *comparable* to exact on angular data, so we flag
    each cell's metric space (design section 9.2)."""
    out.append(f"## {decision}: {exact} vs {approx}\n")
    out.append(f"_{regime_note}_\n")
    out.append(f"| cell | metric | {exact} ms | {approx} ms | time ratio | "
               f"{exact} ARI | {approx} ARI | dARI |")
    out.append("|---|---|---|---|---|---|---|---|")
    for cid in sorted(agg):
        e_ms = agg[cid].get(exact, {}).get("ordering_ms")
        a_ms = agg[cid].get(approx, {}).get("ordering_ms")
        e_ari = agg[cid].get(exact, {}).get("ari")
        a_ari = agg[cid].get(approx, {}).get("ari")
        ratio = (e_ms / a_ms) if (e_ms and a_ms and a_ms > 0) else None
        dari = (a_ari - e_ari) if (a_ari is not None and e_ari is not None
                                   and not np.isnan(a_ari) and not np.isnan(e_ari)) else None
        out.append(f"| {cid} | {cells[cid]['metric_space']} | {_fmt(e_ms,1)} | {_fmt(a_ms,1)} | "
                   f"{(_fmt(ratio,2)+'x') if ratio else 'n/a'} | {_fmt(e_ari)} | {_fmt(a_ari)} | "
                   f"{_fmt(dari) if dari is not None else 'n/a'} |")
    out.append("")


def readiness(cells, agg, out):
    """Report which decisions the current run can resolve vs which need an unswept axis."""
    out.append("## Decision readiness\n")
    dims = sorted({int(cells[c]["d"]) for c in cells})
    ns = sorted({int(cells[c]["n"]) for c in cells})
    densities = sorted({cells[c]["density"] for c in cells})
    have_optics = any("ours-optics" in agg[c] for c in agg)
    have_soptics = any("ours-soptics" in agg[c] for c in agg)
    have_hdb = any("ours-hdbscan" in agg[c] for c in agg)
    have_shdb = any("ours-shdbscan" in agg[c] for c in agg)
    msgs = [
        ("D1 backend by dim", "insufficient -- needs the backend axis (exact/approx/Boost/HNSW); "
         "this run used the exact backend only"),
        ("D2 sOPTICS vs OPTICS",
         "PARTIAL -- see table above" if (have_optics and have_soptics and len(ns) > 1)
         else "needs an n-sweep at fixed (d, density) -- run the 'scaling' tier"),
        ("D3 Precompute vs OnDemand", "insufficient -- needs the mode axis; this run used OnDemand only"),
        ("D4 eps estimator", "insufficient -- needs uniform vs knee; this run used knee only"),
        ("D5 HDBSCAN* vs sHDBSCAN",
         "PARTIAL -- see table above" if (have_hdb and have_shdb and len(ns) > 1)
         else "needs an n-sweep at fixed (d, density) -- run the 'scaling' tier"),
    ]
    out.append(f"Run covers: dims={dims}, n={ns}, densities={densities}.\n")
    for name, msg in msgs:
        out.append(f"- **{name}** -- {msg}")
    out.append("")


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze the benchmark-matrix tidy CSV.")
    p.add_argument("results_csv")
    p.add_argument("--out", default=None, help="also write the Markdown report to this file")
    args = p.parse_args(argv)

    cells, agg = load(args.results_csv)
    if not agg:
        print("no usable rows in", args.results_csv, file=sys.stderr)
        return 2

    out = [f"# Benchmark matrix analysis -- `{args.results_csv}`\n",
           f"{len(cells)} cells, engines: {', '.join(_engines(agg))}\n"]
    quality_table(cells, agg, out)
    speedup_table(cells, agg, out)
    crossover_table(cells, agg, out, "ours-optics", "ours-soptics", "D2",
                    "sOPTICS (cosine) is comparable to exact only on angular/cosine cells.")
    crossover_table(cells, agg, out, "ours-hdbscan", "ours-shdbscan", "D5",
                    "sHDBSCAN (cosine) is comparable to exact only on angular/cosine cells.")
    readiness(cells, agg, out)

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n(report written to {args.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
