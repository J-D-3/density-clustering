"""Aggregate results.csv + runs.json into the tables the REPORT.md is built from.

Pure stdlib (no pandas/sklearn). Writes out/summary.md and prints the same to stdout.
Each section maps to a study hypothesis (H1..H6) so the report can cite it directly.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # console may default to cp1252 on Windows
except Exception:
    pass

OUT = Path(__file__).resolve().parent / "out"
CONFIGS = ["optics-thr", "optics-xi05", "optics-xi01", "hdbscan-sm", "hdbscan-lg", "shdbscan", "soptics"]


def load():
    with open(OUT / "results.csv") as f:
        rows = [r for r in csv.DictReader(f)]
    runs = json.loads((OUT / "runs.json").read_text())
    for r in rows:
        for k in ("n", "unique", "n_clusters", "hits", "n_roles"):
            r[k] = int(float(r[k])) if r[k] not in ("", None) else 0
        for k in ("collapse", "noise_frac", "recall", "top_frac", "t_dedup_ms",
                  "t_core_ms", "t_extract_ms", "wall_s"):
            r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
        r["ok"] = r["ok"] in ("1", 1, "True")
    return rows, runs


def fmt(x, d=2):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def mean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def median(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


def section(title):
    return f"\n## {title}\n"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def main():
    rows, runs = load()
    ok = [r for r in rows if r["ok"]]
    L = []  # markdown lines

    L.append("# Color-clustering study — aggregated results\n")
    L.append(f"Runs: {len(rows)} total, {len(ok)} ok, {len(rows)-len(ok)} failed. "
             f"max_dim={runs.get('max_dim')}.\n")

    # --- H2/H4: recall by config x space x kind -----------------------------
    L.append(section("H2/H4 — Recall of expected colors (mean), by algorithm × space × kind"))
    by = defaultdict(list)
    for r in ok:
        by[(r["config"], r["space"], r["kind"])].append(r["recall"])
    hdr = ["config", "rgb·orig", "rgb·prep", "lab·orig", "lab·prep", "overall"]
    trows = []
    for cfg in CONFIGS:
        cells = [mean(by[(cfg, s, k)]) for s in ("rgb", "lab") for k in ("original", "preprocessed")]
        overall = mean([v for v in (by[(cfg, s, k)] for s in ("rgb", "lab")
                                    for k in ("original", "preprocessed")) for v in v])
        trows.append([cfg] + [fmt(c) for c in cells] + [fmt(overall)])
    L.append(table(hdr, trows))
    L.append("\n_Space effect (Lab − RGB, mean recall over all ok runs):_ "
             f"**{fmt(mean([r['recall'] for r in ok if r['space']=='lab']) - mean([r['recall'] for r in ok if r['space']=='rgb']), 3)}**\n")

    # --- H1: dominant cluster (background) -----------------------------------
    L.append(section("H1 — Dominant cluster color & size, original vs preprocessed (hdbscan-lg)"))
    hdr = ["article", "kind", "top_color (mode)", "mean top_frac", "black-dominant?"]
    trows = []
    grp = defaultdict(list)
    for r in ok:
        if r["config"] == "hdbscan-lg":
            grp[(r["article"], r["kind"])].append(r)
    for (art, kind), rs in sorted(grp.items()):
        tops = [r["top_color"] for r in rs]
        mode = max(set(tops), key=tops.count)
        tf = mean([r["top_frac"] for r in rs])
        black = sum(1 for r in rs if r["top_color"] == "black")
        trows.append([art, kind, mode, fmt(tf), f"{black}/{len(rs)}"])
    L.append(table(hdr, trows))

    # --- H3: preprocessing effect (collapse, runtime, cleanliness) -----------
    L.append(section("H3 — Preprocessing effect (mean over all ok runs of an image kind)"))
    hdr = ["metric", "original", "preprocessed"]
    def mk(metric, key, d=1):
        o = mean([r[key] for r in ok if r["kind"] == "original"])
        p = mean([r[key] for r in ok if r["kind"] == "preprocessed"])
        return [metric, fmt(o, d), fmt(p, d)]
    trows = [
        mk("dedup collapse (×)", "collapse"),
        mk("OPTICS/MST core (ms)", "t_core_ms", 0),
        mk("n_clusters", "n_clusters", 1),
        mk("recall", "recall", 3),
        mk("noise frac", "noise_frac", 3),
    ]
    L.append(table(hdr, trows))

    # --- Timing by algorithm -------------------------------------------------
    L.append(section("Timing — median core (ms) and wall (s) by algorithm"))
    hdr = ["config", "median core ms", "median wall s", "max wall s"]
    trows = []
    for cfg in CONFIGS:
        cc = [r["t_core_ms"] for r in ok if r["config"] == cfg]
        ww = [r["wall_s"] for r in ok if r["config"] == cfg]
        trows.append([cfg, fmt(median(cc), 0), fmt(median(ww), 2),
                      fmt(max(ww) if ww else float('nan'), 1)])
    L.append(table(hdr, trows))

    # --- H6: approximate vs exact agreement (ARI) ---------------------------
    L.append(section("H6 — Approximate vs exact agreement (mean ARI / NMI)"))
    ag = defaultdict(lambda: {"ari": [], "nmi": []})
    want = {("shdbscan", "hdbscan-sm"): "shdbscan vs hdbscan-sm",
            ("hdbscan-sm", "shdbscan"): "shdbscan vs hdbscan-sm",
            ("soptics", "optics-thr"): "soptics vs optics-thr",
            ("optics-thr", "soptics"): "soptics vs optics-thr"}
    for d in runs["detail"]:
        if "agreement" in d:
            pair = tuple(d["agreement"])
            if pair in want:
                key = (want[pair], d["kind"])
                ag[key]["ari"].append(d["ari"]); ag[key]["nmi"].append(d["nmi"])
    hdr = ["pair", "kind", "mean ARI", "mean NMI", "n"]
    trows = []
    for (pair, kind), v in sorted(ag.items()):
        trows.append([pair, kind, fmt(mean(v["ari"]), 3), fmt(mean(v["nmi"]), 3), len(v["ari"])])
    L.append(table(hdr, trows))

    # --- H5: same-article drift ---------------------------------------------
    L.append(section("H5 — Same-article drift (cluster-count spread & mean matched ΔE across an article's images)"))
    hdr = ["article", "kind", "space", "config", "n_imgs", "counts", "Δcount", "mean ΔE"]
    trows = []
    for d in sorted(runs["drift"], key=lambda d: (d["article"], d["kind"], d["space"], d["config"])):
        if d["config"] not in ("hdbscan-lg", "optics-thr"):
            continue  # keep the table readable: two representative extractors
        trows.append([d["article"], d["kind"], d["space"], d["config"], d["n_images"],
                      str(d["cluster_counts"]), d["count_spread"], fmt(d["mean_pair_dE"], 1)])
    L.append(table(hdr, trows))

    # --- voxel sub-study -----------------------------------------------------
    if runs.get("voxel"):
        L.append(section("Voxel sub-study — collapse / cluster count / recall / core-time vs bin"))
        hdr = ["article", "space", "algo", "bin", "collapse", "k", "recall", "core ms"]
        trows = []
        for v in runs["voxel"]:
            trows.append([v["article"], v["space"], v["algo"], v["voxel"],
                          fmt(v["collapse"], 1), v["n_clusters"], fmt(v["recall"], 2),
                          fmt(v["t_core_ms"], 0)])
        L.append(table(hdr, trows))

    md = "\n".join(L)
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n-> {OUT/'summary.md'}")


if __name__ == "__main__":
    main()
