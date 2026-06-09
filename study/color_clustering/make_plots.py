"""Render plots for the color-clustering study from runs.json.

Two kinds, both saved under out/plots/:
  * reachability profiles for the OPTICS runs (the cluster-ordering curve -- valleys
    are clusters), one panel per (space) for a curated image set;
  * 3-D RGB color-space scatters colored by each cluster's mean color, for the
    RGB-space runs (where the clustering coordinates ARE the displayable RGB).

Curated to ~one image per article (plus a couple original/preprocessed pairs to
show what background removal does), so the plot set stays readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import study_lib as S

PLOTS = S.OUT / "plots"


def load_runs() -> dict:
    return json.loads((S.OUT / "runs.json").read_text())


def reach_curve(reach_file: str):
    arr = np.genfromtxt(reach_file, delimiter=",", skip_header=1, dtype=float)
    r = arr[:, 2].copy()
    r[r < 0] = np.nan          # UNDEFINED -> gap in the curve
    return r


def plot_reachability(detail, image_stems):
    runs = [d for d in detail if d.get("config") == "optics-thr"
            and d.get("reach_file") and Path(d["reach_file"]).exists()
            and d["image"] in image_stems]
    if not runs:
        return
    by_img = {}
    for d in runs:
        by_img.setdefault(d["image"], {})[d["space"]] = d
    n = len(by_img)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.4 * n), squeeze=False)
    for row, (img, spaces) in enumerate(sorted(by_img.items())):
        for col, space in enumerate(("rgb", "lab")):
            ax = axes[row][col]
            d = spaces.get(space)
            if d:
                r = reach_curve(d["reach_file"])
                ax.fill_between(range(len(r)), 0, r, step="mid", color="#3b6", alpha=.6)
                ax.set_title(f"{img}  [{space}]  k={d['metrics']['n_clusters']} "
                             f"recall={d['recall']['recall']:.2f}", fontsize=8)
            ax.set_ylabel("reach"); ax.set_xlabel("cluster ordering")
    fig.suptitle("OPTICS reachability (valleys = clusters)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "reachability.png", dpi=110)
    plt.close(fig)
    print("  wrote reachability.png")


def plot_color_scatter(detail, image_stems, configs=("optics-thr", "hdbscan-sm")):
    # RGB-space runs only (coords are displayable RGB).
    sel = [d for d in detail if d.get("space") == "rgb" and d.get("config") in configs
           and d.get("labels_file") and Path(d["labels_file"]).exists()
           and d["image"] in image_stems]
    by_img = {}
    for d in sel:
        by_img.setdefault(d["image"], {})[d["config"]] = d
    for img, cfgs in sorted(by_img.items()):
        fig = plt.figure(figsize=(6 * len(configs), 5.5))
        for i, cfg in enumerate(configs):
            d = cfgs.get(cfg)
            ax = fig.add_subplot(1, len(configs), i + 1, projection="3d")
            if not d:
                continue
            arr = np.genfromtxt(d["labels_file"], delimiter=",", skip_header=1, dtype=float)
            rgb, lab = arr[:, :3], arr[:, 3].astype(int)
            # subsample for speed
            if len(rgb) > 8000:
                idx = np.random.RandomState(0).choice(len(rgb), 8000, replace=False)
                rgb, lab = rgb[idx], lab[idx]
            colors = np.full((len(rgb), 3), 0.6)
            for c in set(lab.tolist()):
                if c < 0:
                    continue
                m = lab == c
                colors[m] = rgb[m].mean(axis=0) / 255.0
            noise = lab < 0
            ax.scatter(rgb[~noise, 0], rgb[~noise, 1], rgb[~noise, 2],
                       c=colors[~noise], s=6, depthshade=False)
            ax.set_title(f"{cfg}  k={d['metrics']['n_clusters']} "
                         f"recall={d['recall']['recall']:.2f}", fontsize=9)
            ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
        fig.suptitle(f"{img}: RGB color space colored by cluster mean", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOTS / f"scatter_{img}.png", dpi=100)
        plt.close(fig)
        print(f"  wrote scatter_{img}.png")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    detail = [d for d in runs["detail"] if "config" in d]
    # curated image set: one representative per article + a couple prepped pairs
    want = ["Amlodipin_5mg_01", "Amlodipin_05_01_prepped",
            "Spiegel_ChatGPT_01", "Spiegel_ChatGPT_01_prepped",
            "Spiegel_Hoffnung_01", "Spiegel_Mutante_01", "Bild_Koeln_01"]
    stems = set(d["image"] for d in detail)
    image_stems = [w for w in want if w in stems] or sorted(stems)[:6]
    print("plotting for:", image_stems)
    plot_reachability(detail, image_stems)
    plot_color_scatter(detail, image_stems)
    print(f"  -> {PLOTS}")


if __name__ == "__main__":
    main()
