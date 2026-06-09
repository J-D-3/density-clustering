"""Drive the color-clustering study matrix and write results.

Matrix = (image) x (color space: RGB, Lab) x (algorithm config). For each cell it
invokes the C++ `color_study` harness, parses its JSON metrics, computes a per-cluster
color summary, scores recall of the expected color roles, and records everything.
After each (image, space) cell it also computes inter-algorithm agreement (ARI/NMI),
and at the end it computes same-article cluster drift.

Outputs (under out/):
  results.csv  -- one row per run (flat, for quick scanning / spreadsheets)
  runs.json    -- full detail: per-cluster summaries, recall breakdown, agreement, drift
  work/        -- per-run labels + reachability CSVs (gitignored; consumed by make_plots.py)

Usage:
  python run_study.py                      # full matrix
  python run_study.py --quick amlodipin_5mg  # one article only (dry-run)
  python run_study.py --max-dim 320 --timeout 180
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

import study_lib as S


def density_params(n: int) -> dict:
    """Size-relative density knobs so a config means the same thing across image sizes."""
    return {
        "min_pts": max(8, round(0.0007 * n)),
        "mcs_small": max(15, round(0.002 * n)),
        "mcs_large": max(40, round(0.010 * n)),
    }


def configs_for(n: int) -> list[dict]:
    """The algorithm configs run on every image. `flags` are passed to the harness."""
    d = density_params(n)
    mp, frac = d["min_pts"], 0.003
    return [
        {"name": "optics-thr",  "algo": "optics-threshold",
         "flags": {"min_pts": mp, "threshold": -1, "min_cluster_frac": frac}},
        {"name": "optics-xi05", "algo": "optics-xi",
         "flags": {"min_pts": mp, "xi": 0.05, "min_cluster_frac": frac}},
        {"name": "optics-xi01", "algo": "optics-xi",
         "flags": {"min_pts": mp, "xi": 0.01, "min_cluster_frac": frac}},
        {"name": "hdbscan-sm",  "algo": "hdbscan",
         "flags": {"min_cluster_size": d["mcs_small"], "min_cluster_frac": frac}},
        {"name": "hdbscan-lg",  "algo": "hdbscan",
         "flags": {"min_cluster_size": d["mcs_large"], "min_cluster_frac": frac}},
        {"name": "shdbscan",    "algo": "shdbscan",
         "flags": {"min_cluster_size": d["mcs_small"], "metric": "l2", "seed": 42,
                   "n_projections": 512, "min_cluster_frac": frac}},
        {"name": "soptics",     "algo": "soptics",
         "flags": {"min_pts": mp, "threshold": -1, "metric": "l2", "seed": 42,
                   "n_projections": 512, "min_cluster_frac": frac}},
    ]


SPACES = ["rgb", "lab"]


def run_matrix(entries, max_dim, timeout, work: Path):
    work.mkdir(parents=True, exist_ok=True)
    rows = []          # flat result rows
    detail = []        # rich per-run detail (for runs.json)
    # keep per-(article,kind,space,config) cluster summaries for the drift analysis
    by_group: dict[tuple, list] = {}

    for e in entries:
        pixels = S.load_pixels(e.path, max_dim=max_dim)
        n = len(pixels)
        lab_all = S.srgb_to_lab(pixels)
        spaces = {"rgb": pixels.astype(float), "lab": lab_all}
        print(f"\n## {e.label}  ({e.article}/{e.kind})  n={n}")
        for space in SPACES:
            csv_path = work / f"{e.path.stem}_{e.kind}_{space}.csv"
            S.write_csv(spaces[space], csv_path)
            labels_by_cfg = {}
            for cfg in configs_for(n):
                lab_out = work / f"{e.path.stem}_{e.kind}_{space}_{cfg['name']}.labels.csv"
                reach_out = (work / f"{e.path.stem}_{e.kind}_{space}_{cfg['name']}.reach.csv"
                             if cfg["algo"] in ("optics-threshold", "optics-xi", "soptics") else None)
                t0 = time.time()
                res = S.run_harness(csv_path, cfg["algo"], labels_out=lab_out,
                                    reach_out=reach_out, timeout=timeout, **cfg["flags"])
                wall = time.time() - t0
                if not res.ok:
                    print(f"  {space:3s} {cfg['name']:11s} FAILED: {res.error}")
                    rows.append({"image": e.path.stem, "article": e.article, "kind": e.kind,
                                 "space": space, "config": cfg["name"], "algo": cfg["algo"],
                                 "n": n, "ok": 0, "error": res.error, "wall_s": round(wall, 2)})
                    continue
                summary = S.cluster_summary(pixels.astype(float), lab_all, res.labels)
                roles = S.expected_roles(e.article, e.kind)
                recall = S.score_recall(summary, roles)
                m = res.metrics
                labels_by_cfg[cfg["name"]] = res.labels
                top = summary[0] if summary else {"name": "-", "frac": 0.0}
                print(f"  {space:3s} {cfg['name']:11s} "
                      f"k={m['n_clusters']:>3} noise={m['noise']/n:4.0%} "
                      f"recall={recall['recall']:.2f} ({recall['hits']}/{recall['n_roles']}) "
                      f"collapse={m['collapse']:.1f}x core={m['t_core_ms']:.0f}ms wall={wall:.1f}s")
                rows.append({
                    "image": e.path.stem, "article": e.article, "kind": e.kind, "space": space,
                    "config": cfg["name"], "algo": cfg["algo"], "n": n,
                    "unique": m["unique"], "collapse": round(m["collapse"], 2),
                    "n_clusters": m["n_clusters"], "noise_frac": round(m["noise"] / n, 4),
                    "top_color": top["name"], "top_frac": round(top["frac"], 3),
                    "recall": round(recall["recall"], 3), "hits": recall["hits"],
                    "n_roles": recall["n_roles"],
                    "t_dedup_ms": round(m["t_dedup_ms"], 1), "t_core_ms": round(m["t_core_ms"], 1),
                    "t_extract_ms": round(m["t_extract_ms"], 1), "wall_s": round(wall, 2),
                    "ok": 1, "error": "",
                })
                detail.append({
                    "image": e.path.stem, "article": e.article, "kind": e.kind, "space": space,
                    "config": cfg["name"], "algo": cfg["algo"], "metrics": m,
                    "summary": summary[:12], "recall": recall,
                    "labels_file": str(lab_out),
                    "reach_file": str(reach_out) if reach_out else None,
                })
                by_group.setdefault((e.article, e.kind, space, cfg["name"]), []).append(summary)

            # inter-algorithm agreement within this (image, space) cell
            cfg_names = list(labels_by_cfg)
            for a, b in itertools.combinations(cfg_names, 2):
                ag = S.agreement(labels_by_cfg[a], labels_by_cfg[b])
                if ag:
                    detail.append({"image": e.path.stem, "article": e.article, "kind": e.kind,
                                   "space": space, "agreement": [a, b], **ag})

    # same-article drift: across the images of one (article, kind, space, config)
    drift = []
    for (article, kind, space, cfg), summaries in sorted(by_group.items()):
        if len(summaries) < 2:
            continue
        pairs = [S.match_clusters_across(summaries[i], summaries[j])
                 for i in range(len(summaries)) for j in range(i + 1, len(summaries))]
        ks = [len(s) for s in summaries]
        drift.append({
            "article": article, "kind": kind, "space": space, "config": cfg,
            "n_images": len(summaries), "cluster_counts": ks,
            "count_spread": max(ks) - min(ks),
            "mean_pair_dE": round(float(np.mean([p["mean_matched_dE"] for p in pairs
                                                 if p["mean_matched_dE"] is not None])), 1),
        })
    return rows, detail, drift


def voxel_substudy(max_dim, timeout, work: Path):
    """Isolate the voxel knob on two dense images: collapse, runtime, cluster stability vs bin."""
    print("\n## voxel sub-study")
    out = []
    targets = [e for e in S.build_manifest()
               if e.kind == "original" and e.article in ("spiegel_mutante", "bild_koeln")
               and e.path.stem.endswith("_01")]
    grids = {"rgb": [0, 4, 8, 16], "lab": [0, 2, 4, 8]}
    for e in targets:
        pixels = S.load_pixels(e.path, max_dim=max_dim)
        n = len(pixels)
        lab_all = S.srgb_to_lab(pixels)
        spaces = {"rgb": pixels.astype(float), "lab": lab_all}
        for space in ("rgb", "lab"):
            csv_path = work / f"voxel_{e.path.stem}_{space}.csv"
            S.write_csv(spaces[space], csv_path)
            for algo, name in (("optics-threshold", "optics-thr"), ("hdbscan", "hdbscan-sm")):
                d = density_params(n)
                base = ({"min_pts": d["min_pts"], "threshold": -1} if algo == "optics-threshold"
                        else {"min_cluster_size": d["mcs_small"]})
                for bin_ in grids[space]:
                    lab_out = work / f"voxel_{e.path.stem}_{space}_{name}_b{bin_}.labels.csv"
                    res = S.run_harness(csv_path, algo, labels_out=lab_out, timeout=timeout,
                                        voxel=bin_, min_cluster_frac=0.003, **base)
                    if not res.ok:
                        print(f"  {e.article:15s} {space} {name} bin={bin_:2} FAILED {res.error}")
                        continue
                    summary = S.cluster_summary(pixels.astype(float), lab_all, res.labels)
                    recall = S.score_recall(summary, S.expected_roles(e.article, e.kind))
                    m = res.metrics
                    print(f"  {e.article:15s} {space} {name} bin={bin_:2} "
                          f"collapse={m['collapse']:5.1f}x k={m['n_clusters']:>2} "
                          f"recall={recall['recall']:.2f} core={m['t_core_ms']:.0f}ms")
                    out.append({"image": e.path.stem, "article": e.article, "space": space,
                                "algo": name, "voxel": bin_, "n": n, "unique": m["unique"],
                                "collapse": round(m["collapse"], 2), "n_clusters": m["n_clusters"],
                                "recall": round(recall["recall"], 3),
                                "t_core_ms": round(m["t_core_ms"], 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", default=None, help="run only this article (dry-run)")
    ap.add_argument("--max-dim", type=int, default=320)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--no-voxel", action="store_true")
    args = ap.parse_args()

    if not S.HARNESS.exists():
        raise SystemExit(f"harness not built: {S.HARNESS}")

    entries = S.build_manifest()
    if args.quick:
        entries = [e for e in entries if e.article == args.quick]
        if not entries:
            raise SystemExit(f"no images for article '{args.quick}'")

    work = S.OUT / "work"
    t0 = time.time()
    rows, detail, drift = run_matrix(entries, args.max_dim, args.timeout, work)
    voxel = [] if args.no_voxel or args.quick else voxel_substudy(args.max_dim, args.timeout, work)

    S.OUT.mkdir(parents=True, exist_ok=True)
    # results.csv (union of keys, stable column order)
    cols = ["image", "article", "kind", "space", "config", "algo", "n", "unique", "collapse",
            "n_clusters", "noise_frac", "top_color", "top_frac", "recall", "hits", "n_roles",
            "t_dedup_ms", "t_core_ms", "t_extract_ms", "wall_s", "ok", "error"]
    with open(S.OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    with open(S.OUT / "runs.json", "w") as f:
        json.dump({"detail": detail, "drift": drift, "voxel": voxel,
                   "max_dim": args.max_dim}, f, indent=1)
    if voxel:
        with open(S.OUT / "voxel.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(voxel[0].keys()))
            w.writeheader(); w.writerows(voxel)

    ok = sum(r["ok"] for r in rows)
    print(f"\nDONE: {ok}/{len(rows)} runs ok, {len(drift)} drift groups, "
          f"{len(voxel)} voxel runs, {time.time()-t0:.0f}s total")
    print(f"  -> {S.OUT/'results.csv'}\n  -> {S.OUT/'runs.json'}")


if __name__ == "__main__":
    main()
