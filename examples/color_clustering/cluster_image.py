#!/usr/bin/env python3
"""Cluster an image's colors with the OPTICS library and plot the 3D color space.

Pipeline: load image -> downscale -> RGB pixels -> CSV -> optics_color (the C++
library tool) -> labeled CSV -> 3D scatter (colored by cluster, or by true RGB
with --rgb).

Example:
  python tools/cluster_image.py hexal.jpg --exe build/optics_color.exe \
      --max-dim 240 --min-pts 25 --threshold 30 --out hexal_clusters.png
"""
import argparse
import csv as csvmod
import subprocess
import sys

import numpy as np
from PIL import Image


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--exe", required=True, help="path to the optics_color executable")
    ap.add_argument("--max-dim", type=int, default=240, help="downscale so the longest side is <= this")
    ap.add_argument("--min-pts", type=int, default=25)
    ap.add_argument("--eps", type=float, default=-1.0, help="<=0 auto-estimates")
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--frac", type=float, default=0.01, help="min cluster size as a fraction of pixels")
    ap.add_argument("--out", default="color_clusters.png")
    ap.add_argument("--rgb", action="store_true", help="color points by their true RGB instead of by cluster")
    ap.add_argument("--plot-sample", type=int, default=8000, help="max points to scatter (for legibility)")
    args = ap.parse_args(argv)

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    scale = min(1.0, args.max_dim / max(w, h))
    if scale < 1.0:
        # NEAREST preserves true pixel colors (BILINEAR would invent blend colors
        # at edges, which then form their own clusters in color space).
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
    pixels = np.asarray(img).reshape(-1, 3)
    print(f"image {args.image}: {w}x{h} -> {img.size[0]}x{img.size[1]} = {len(pixels)} pixels")

    in_csv, out_csv = "colors_in.csv", "colors_clustered.csv"
    np.savetxt(in_csv, pixels, fmt="%d", delimiter=",")
    subprocess.run([args.exe, in_csv, out_csv, str(args.min_pts), str(args.eps),
                    str(args.threshold), str(args.frac)], check=True)
    render(out_csv, args.out, args.rgb, args.plot_sample)
    return 0


def render(csv_path, out, by_rgb, sample):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords, labels = [], []
    with open(csv_path, newline="") as f:
        r = csvmod.reader(f)
        next(r)  # header
        for row in r:
            coords.append([float(row[0]), float(row[1]), float(row[2])])
            labels.append(int(row[3]))
    coords = np.asarray(coords)
    labels = np.asarray(labels)

    # Subsample for a legible scatter (color clouds have many duplicate points).
    if len(coords) > sample:
        idx = np.random.default_rng(0).choice(len(coords), sample, replace=False)
        coords, labels = coords[idx], labels[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if by_rgb:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=coords / 255.0, s=5)
        title = "Color space (true colors)"
    else:
        # noise = -1 drawn light gray; clusters get distinct colors.
        noise = labels < 0
        ax.scatter(coords[noise, 0], coords[noise, 1], coords[noise, 2], c="lightgray", s=3, alpha=0.3)
        cl = ~noise
        ax.scatter(coords[cl, 0], coords[cl, 1], coords[cl, 2], c=labels[cl], cmap="tab10", s=6)
        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        title = f"Clustered color space ({n_clusters} clusters)"
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
