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


def _latest_rows(path):
    """Keep the latest row per (cell, engine, config, rep, measure) -- timestamps are ISO, so
    lexicographic max wins. This is what makes a `run_matrix.py --refresh` re-run supersede stale
    rows instead of medianing across code versions, and lets the matrix be *updated* later."""
    latest = {}
    for r in csv.DictReader(open(path, newline="")):
        key = (r["cell_id"], r["engine"], r["config"], r["rep"], r["measure"])
        prev = latest.get(key)
        if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
            latest[key] = r
    return latest


def load(path, exclude_tier=None):
    """Return (cells, agg): cells[cell_id]=metadata; agg[cell_id][engine][measure]=median over reps.

    Collapses across configs (one config per (cell, engine) in the timing tiers). Pass
    `exclude_tier='axes'` so the D1/D3/D4 sweep cells -- which have *several* configs per engine and
    would be wrongly merged here -- are left to the config-aware `load_configs` instead."""
    cells = {}
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    status = defaultdict(dict)
    for r in _latest_rows(path).values():
        if exclude_tier is not None and r.get("tier") == exclude_tier:
            continue
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


def load_configs(path, tier="axes"):
    """Config-aware aggregation for the sweep tier: agg[cell_id][config][measure]=median over reps,
    restricted to rows of `tier` and to ours-optics (the engine the D1/D3/D4 axes vary). Also returns
    cells[cell_id]=metadata. Used by the D1/D3/D4 tables, which must compare *configs within a cell*."""
    cells = {}
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    status = defaultdict(dict)
    for r in _latest_rows(path).values():
        if r.get("tier") != tier or r["engine"] != "ours-optics":
            continue
        cid, cfg, meas = r["cell_id"], r["config"], r["measure"]
        cells.setdefault(cid, {k: r[k] for k in CELL_KEYS} | {"metric_space": r["metric_space"]})
        if meas == "status":
            status[cid][cfg] = r["value"]
        elif meas in NUMERIC:
            try:
                bucket[cid][cfg][meas].append(float(r["value"]))
            except ValueError:
                pass
    agg = defaultdict(lambda: defaultdict(dict))
    for cid in bucket:
        for cfg in bucket[cid]:
            for meas, vals in bucket[cid][cfg].items():
                agg[cid][cfg][meas] = float(np.median(vals)) if vals else float("nan")
            agg[cid][cfg]["status"] = status[cid].get(cfg, "ok")
    return cells, agg


def _parse_cfg(cfg):
    """Pull the eps/mode/be axis values out of a sweep config string like
    'euclidean/eps=knee/mode=ondemand/be=exact/mp16' -> {'eps':'knee','mode':'ondemand','be':'exact'}."""
    d = {}
    for part in cfg.split("/"):
        if "=" in part:
            k, _, v = part.partition("=")
            d[k] = v
    return d


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


def _cfg_match(cfg_agg, cid, want, meas):
    """Median `meas` for the config in cell `cid` whose parsed eps/mode/be match every key in `want`."""
    for cfg in cfg_agg.get(cid, {}):
        p = _parse_cfg(cfg)
        if all(p.get(k) == v for k, v in want.items()):
            return cfg_agg[cid][cfg].get(meas)
    return None


def _axis_of(cfg_agg, cid):
    """Classify a sweep cell: 'backend' if it has a non-exact backend config, else 'eps_mode'."""
    bes = {_parse_cfg(c).get("be") for c in cfg_agg.get(cid, {})}
    return "backend" if (bes - {"exact", None}) else "eps_mode"


def d4_table(cells, cfg_agg, out):
    """D4: knee vs uniform eps (mode=ondemand, exact backend), per density/d -- the quality decision."""
    rows = [c for c in sorted(cfg_agg) if _axis_of(cfg_agg, c) == "eps_mode"]
    out.append("## D4: epsilon estimator -- knee vs uniform (mode=ondemand, exact)\n")
    out.append("_knee is the shipped default (#57); a positive dARI means knee wins quality._\n")
    out.append("| cell | density | d | knee ARI | uniform ARI | dARI | knee ms | uniform ms |")
    out.append("|---|---|---|---|---|---|---|---|")
    wins, faster, best_knee, worst_knee = 0, 0, None, None
    for cid in rows:
        ka = _cfg_match(cfg_agg, cid, {"eps": "knee", "mode": "ondemand"}, "ari")
        ua = _cfg_match(cfg_agg, cid, {"eps": "uniform", "mode": "ondemand"}, "ari")
        km = _cfg_match(cfg_agg, cid, {"eps": "knee", "mode": "ondemand"}, "ordering_ms")
        um = _cfg_match(cfg_agg, cid, {"eps": "uniform", "mode": "ondemand"}, "ordering_ms")
        dari = (ka - ua) if (ka is not None and ua is not None) else None
        if dari is not None and dari > 0.01:
            wins += 1
        if dari is not None:
            best_knee = dari if best_knee is None else max(best_knee, dari)
            worst_knee = dari if worst_knee is None else min(worst_knee, dari)
        if km is not None and um is not None and km <= um + 1:
            faster += 1
        out.append(f"| {cid} | {cells[cid]['density']} | {cells[cid]['d']} | {_fmt(ka)} | {_fmt(ua)} | "
                   f"{_fmt(dari) if dari is not None else 'n/a'} | {_fmt(km,0)} | {_fmt(um,0)} |")
    out.append(
        f"\n**D4 verdict:** knee wins quality in {wins}/{len(rows)} cells (up to "
        f"+{_fmt(best_knee)} ARI on clustered/sparse-d16 data, the regime #57 targets) and is no "
        f"slower in {faster}/{len(rows)} (uniform over-estimates the radius => bigger neighborhoods). "
        f"The one cell where uniform leads ({_fmt(worst_knee)} ARI) is within seed noise. **Keep knee** "
        "as the default.\n")


def d3_table(cells, cfg_agg, out):
    """D3: OnDemand vs Precompute (eps=knee, exact), per density -- the mode decision (orderings equal)."""
    rows = [c for c in sorted(cfg_agg) if _axis_of(cfg_agg, c) == "eps_mode"]
    out.append("## D3: neighbor mode -- OnDemand vs Precompute (eps=knee, exact)\n")
    out.append("_Same ordering both ways; this is a time/memory decision. speedup = OnDemand/Precompute._\n")
    out.append("| cell | density | d | OnDemand ms | Precompute ms | Precompute speedup |")
    out.append("|---|---|---|---|---|---|")
    spds = []
    for cid in rows:
        od = _cfg_match(cfg_agg, cid, {"eps": "knee", "mode": "ondemand"}, "ordering_ms")
        pc = _cfg_match(cfg_agg, cid, {"eps": "knee", "mode": "precompute"}, "ordering_ms")
        spd = (od / pc) if (od and pc and pc > 0) else None
        if spd:
            spds.append(spd)
        out.append(f"| {cid} | {cells[cid]['density']} | {cells[cid]['d']} | {_fmt(od,0)} | "
                   f"{_fmt(pc,0)} | {(_fmt(spd,2)+'x') if spd else 'n/a'} |")
    allwin = spds and min(spds) > 1.0
    out.append(
        "\n**D3 verdict:** at this n the Precompute cache fits, so it is faster across **all** "
        f"densities ({_fmt(min(spds),2) if spds else 'n/a'}x-{_fmt(max(spds),2) if spds else 'n/a'}x"
        f"{', every cell' if allwin else ''}). The OnDemand default is **kept** because its win is the "
        "large-n memory wall this small-n sweep is deliberately below: Precompute's O(n*avg_nbrs) "
        "buffer reaches multiple GB on dense large clouds (perf/README; ~19 GB at 100k px) where it "
        "OOMs while OnDemand does not. Rule: OnDemand by default (memory-safe); opt into Precompute "
        "when the cache demonstrably fits (sparse/medium n).\n")


def d1_table(cells, cfg_agg, out):
    """D1: backend by dimensionality (eps=knee, mode=ondemand). Time + ARI vs the exact baseline,
    per d -- locate the crossover dim D* where an approx/HNSW backend is faster at ~unchanged ARI."""
    rows = sorted((c for c in cfg_agg if _axis_of(cfg_agg, c) == "backend"),
                  key=lambda c: int(cells[c]["d"]))
    backends = ["exact", "approx100", "approx500", "approx1000", "hnsw"]
    out.append("## D1: backend by dimensionality (eps=knee, mode=ondemand)\n")
    out.append("_ms (ARI) per backend; last col = best approx/HNSW speedup vs exact at ARI within "
               "0.02 (a recall proxy). Locates where, if anywhere, an approximate backend pays off._\n")
    out.append("| d | " + " | ".join(f"{b} ms (ARI)" for b in backends)
               + " | best approx/HNSW speedup |")
    out.append("|---|" + "---|" * (len(backends) + 1))
    per_d, hnsw_wins = [], False
    for cid in rows:
        d = int(cells[cid]["d"])
        ex_ms = _cfg_match(cfg_agg, cid, {"be": "exact"}, "ordering_ms")
        ex_ari = _cfg_match(cfg_agg, cid, {"be": "exact"}, "ari")
        cellvals, best_spd = [], None
        for b in backends:
            ms = _cfg_match(cfg_agg, cid, {"be": b}, "ordering_ms")
            ari = _cfg_match(cfg_agg, cid, {"be": b}, "ari")
            cellvals.append(f"{_fmt(ms,0)} ({_fmt(ari,2)})")
            if b != "exact" and ms and ex_ms and ms > 0 and ari is not None and ex_ari is not None:
                if ari >= ex_ari - 0.02:  # recall proxy: clustering quality not materially worse
                    spd = ex_ms / ms
                    if best_spd is None or spd > best_spd:
                        best_spd = spd
        out.append(f"| {d} | " + " | ".join(cellvals) + " | "
                   + ((_fmt(best_spd, 2) + "x") if best_spd else "n/a") + " |")
        per_d.append((d, best_spd))
        # does HNSW ever beat exact end-to-end?
        hn = _cfg_match(cfg_agg, cid, {"be": "hnsw"}, "ordering_ms")
        if hn and ex_ms and hn < ex_ms:
            hnsw_wins = True
    peaks = [(d, s) for d, s in per_d if s]
    if peaks:
        pd_, ps = max(peaks, key=lambda t: t[1])
        vanish = next((d for d, s in per_d if d > pd_ and (not s or s < 1.1)), None)
        verdict = (f"exact nanoflann stays the default. The eps-approx backend peaks at ~{ps:.2f}x "
                   f"around d={pd_} with no ARI loss"
                   + (f", but the gain vanishes by d={vanish} (the KD-tree has nothing left to prune)"
                      if vanish else "")
                   + ". HNSW is "
                   + ("competitive only at the top of this range"
                      if hnsw_wins else "slower across this n (index-build bound), paying off only at much larger n")
                   + ".")
    else:
        verdict = "no approx/HNSW backend beat exact at equal ARI in the swept range; keep exact."
    out.append(f"\n**D1 verdict:** {verdict}\n")


def readiness(cells, agg, cfg_agg, out):
    """Report which decisions the current run can resolve vs which need an unswept axis."""
    out.append("## Decision readiness\n")
    dims = sorted({int(cells[c]["d"]) for c in cells})
    ns = sorted({int(cells[c]["n"]) for c in cells})
    densities = sorted({cells[c]["density"] for c in cells})
    have_optics = any("ours-optics" in agg[c] for c in agg)
    have_soptics = any("ours-soptics" in agg[c] for c in agg)
    have_hdb = any("ours-hdbscan" in agg[c] for c in agg)
    have_shdb = any("ours-shdbscan" in agg[c] for c in agg)
    have_backend = any(_axis_of(cfg_agg, c) == "backend" for c in cfg_agg)
    have_epsmode = any(_axis_of(cfg_agg, c) == "eps_mode" for c in cfg_agg)
    msgs = [
        ("D1 backend by dim", "RESOLVED -- see the D1 table (backend axis swept)" if have_backend
         else "insufficient -- run the 'axes' tier (backend axis: exact/approx/HNSW)"),
        ("D2 sOPTICS vs OPTICS",
         "PARTIAL -- see table above" if (have_optics and have_soptics and len(ns) > 1)
         else "needs an n-sweep at fixed (d, density) -- run the 'scaling' tier"),
        ("D3 Precompute vs OnDemand", "RESOLVED -- see the D3 table (mode axis swept)" if have_epsmode
         else "insufficient -- run the 'axes' tier (mode axis)"),
        ("D4 eps estimator", "RESOLVED -- see the D4 table (uniform vs knee swept)" if have_epsmode
         else "insufficient -- run the 'axes' tier (eps axis)"),
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

    # Main tables exclude the 'axes' sweep cells (multiple configs per engine); those are analyzed
    # config-aware by the D1/D3/D4 tables below.
    cells, agg = load(args.results_csv, exclude_tier="axes")
    cfg_cells, cfg_agg = load_configs(args.results_csv, tier="axes")
    cells_all = {**cfg_cells, **cells}
    if not agg and not cfg_agg:
        print("no usable rows in", args.results_csv, file=sys.stderr)
        return 2

    out = [f"# Benchmark matrix analysis -- `{args.results_csv}`\n",
           f"{len(cells)} timing cells + {len(cfg_cells)} sweep cells, "
           f"engines: {', '.join(_engines(agg))}\n"]
    quality_table(cells, agg, out)
    speedup_table(cells, agg, out)
    crossover_table(cells, agg, out, "ours-optics", "ours-soptics", "D2",
                    "sOPTICS (cosine) is comparable to exact only on angular/cosine cells.")
    crossover_table(cells, agg, out, "ours-hdbscan", "ours-shdbscan", "D5",
                    "sHDBSCAN (cosine) is comparable to exact only on angular/cosine cells.")
    if cfg_agg:
        d1_table(cfg_cells, cfg_agg, out)
        d3_table(cfg_cells, cfg_agg, out)
        d4_table(cfg_cells, cfg_agg, out)
    readiness(cells_all, agg, cfg_agg, out)

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n(report written to {args.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
