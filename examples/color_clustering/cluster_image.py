#!/usr/bin/env python3
"""Cluster an image's colors with the OPTICS library and plot the 3D color space.

Pipeline: load image -> downscale -> RGB pixels -> CSV -> optics_color (the C++
library tool) -> labeled CSV -> 3D scatter. Each cluster's pixels are drawn in
the cluster's mean color, wrapped in a transparent (approx. minimal) enclosing
sphere of the same color.

Output modes:
  --out plot.png   save a static image
  (neither)        open an interactive matplotlib window (rotate: drag, zoom: scroll)
  --html plot.html save an interactive plotly page (requires `pip install plotly`)

Example:
  python cluster_image.py hexal.jpg --exe build/examples/Release/optics_color.exe \
      --min-pts 15 --eps 6 --threshold 12 --frac 0.0008 --max-dim 360
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
    ap.add_argument("--out", help="save a static image to this path (PNG)")
    ap.add_argument("--html", help="save an interactive plotly page to this path (needs plotly)")
    ap.add_argument("--true-color", action="store_true",
                    help="color each pixel by its own RGB instead of by the cluster mean")
    ap.add_argument("--no-spheres", action="store_true", help="do not draw enclosing spheres")
    ap.add_argument("--plot-sample", type=int, default=12000, help="max points to scatter (for legibility)")
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

    coords, labels = load(out_csv)
    if args.html:
        render_plotly(coords, labels, args)
    else:
        render_mpl(coords, labels, args)
    return 0


def load(csv_path):
    coords, labels = [], []
    with open(csv_path, newline="") as f:
        r = csvmod.reader(f)
        next(r)  # header
        for row in r:
            coords.append([float(row[0]), float(row[1]), float(row[2])])
            labels.append(int(row[3]))
    return np.asarray(coords), np.asarray(labels)


def enclosing_sphere(pts):
    """Ritter's approximate minimal enclosing sphere (fast, guaranteed to enclose)."""
    if len(pts) > 4000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
    p = pts[0]
    x = pts[np.argmax(((pts - p) ** 2).sum(1))]
    y = pts[np.argmax(((pts - x) ** 2).sum(1))]
    c = (x + y) / 2.0
    r = float(np.linalg.norm(y - x) / 2.0)
    for q in pts:
        d = float(np.linalg.norm(q - c))
        if d > r:
            r = (r + d) / 2.0
            c = c + (q - c) * (1.0 - r / d)
    return c, r


def cluster_info(coords, labels):
    """Return {label: (mean_rgb, sphere_center, sphere_radius)} for each cluster."""
    info = {}
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    for c in range(n_clusters):
        pts = coords[labels == c]
        if len(pts) == 0:
            continue
        center, radius = enclosing_sphere(pts)
        info[c] = (pts.mean(0), center, radius)
    return info


def render_mpl(coords, labels, args):
    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = cluster_info(coords, labels)

    pc, pl = coords, labels
    if len(coords) > args.plot_sample:
        idx = np.random.default_rng(0).choice(len(coords), args.plot_sample, replace=False)
        pc, pl = coords[idx], labels[idx]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    # A mid-gray backdrop so both near-white and dark points stay visible.
    fig.patch.set_facecolor((0.6, 0.6, 0.6))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.45, 0.45, 0.45, 1.0))

    if args.true_color:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc / 255.0, s=6, depthshade=False)
    else:
        noise = pl < 0
        if noise.any():
            ax.scatter(pc[noise, 0], pc[noise, 1], pc[noise, 2], c=[(0.3, 0.3, 0.3)], s=3, alpha=0.3)
        cl = ~noise
        if cl.any():
            cols = np.array([info[l][0] / 255.0 for l in pl[cl]])
            ax.scatter(pc[cl, 0], pc[cl, 1], pc[cl, 2], c=cols, s=8, depthshade=False)

    if not args.no_spheres:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        for mean, center, radius in info.values():
            sx = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            sy = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            sz = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(sx, sy, sz, color=mean / 255.0, alpha=0.13, linewidth=0, shade=False)

    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
    ax.set_title(f"Clustered color space ({len(info)} clusters)")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120, facecolor=fig.get_facecolor())
        print(f"wrote {args.out}")
    else:
        print("opening interactive window (drag to rotate, scroll to zoom)...")
        plt.show()


def render_plotly(coords, labels, args):
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("--html needs plotly:  pip install plotly")

    info = cluster_info(coords, labels)
    pc, pl = coords, labels
    if len(coords) > args.plot_sample:
        idx = np.random.default_rng(0).choice(len(coords), args.plot_sample, replace=False)
        pc, pl = coords[idx], labels[idx]

    traces = []
    if args.true_color:
        cols = ["rgb(%d,%d,%d)" % tuple(int(x) for x in p) for p in pc]
        traces.append(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                                   marker=dict(size=2, color=cols), name="pixels"))
    else:
        for c, (mean, _, _) in info.items():
            m = pl == c
            col = "rgb(%d,%d,%d)" % tuple(int(x) for x in mean)
            traces.append(go.Scatter3d(x=pc[m, 0], y=pc[m, 1], z=pc[m, 2], mode="markers",
                                       marker=dict(size=2, color=col), name=f"cluster {c}"))

    if not args.no_spheres:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        for mean, center, radius in info.values():
            sx = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            sy = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            sz = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
            col = "rgb(%d,%d,%d)" % tuple(int(x) for x in mean)
            traces.append(go.Surface(x=sx, y=sy, z=sz, opacity=0.15, showscale=False,
                                     colorscale=[[0, col], [1, col]], surfacecolor=np.zeros_like(sx)))

    fig = go.Figure(traces)
    fig.update_layout(scene=dict(xaxis_title="R", yaxis_title="G", zaxis_title="B",
                                 xaxis=dict(range=[0, 255]), yaxis=dict(range=[0, 255]),
                                 zaxis=dict(range=[0, 255])),
                      title=f"Clustered color space ({len(info)} clusters)")
    fig.write_html(args.html)
    print(f"wrote {args.html} (open in a browser; drag to rotate, scroll to zoom)")


if __name__ == "__main__":
    sys.exit(main())
