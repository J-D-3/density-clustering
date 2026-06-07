#!/usr/bin/env python3
"""Time OPTICS (this library) vs k-means (scikit-learn) on an image's colors.

Runs the OPTICS color clustering via the optics_color tool, reads back how many
clusters it found and how long the *ordering* took (excluding CSV I/O), then runs
scikit-learn KMeans with k = that cluster count on the same RGB pixels and times
its fit. This compares the clustering algorithms head to head (OPTICS finds k
automatically; k-means is told k and assigns every point).

Example:
  python compare_kmeans.py Spiegel.jpg --exe build/examples/Release/optics_color.exe \
      --max-dim 240 --min-pts 15 --eps 3 --threshold 7 --frac 0.003
"""
import argparse
import math
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--max-dim", type=int, default=240)
    ap.add_argument("--min-pts", type=int, default=15)
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=7.0)
    ap.add_argument("--frac", type=float, default=0.003)
    args = ap.parse_args(argv)

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    scale = min(1.0, args.max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
    pixels = np.asarray(img).reshape(-1, 3).astype(np.float64)
    n = len(pixels)
    print(f"{args.image}: {n} pixels (downscaled to {img.size[0]}x{img.size[1]})\n")

    # --- OPTICS (this library, via the tool) -------------------------------
    np.savetxt("colors_in.csv", pixels.astype(int), fmt="%d", delimiter=",")
    out = subprocess.run([args.exe, "colors_in.csv", "colors_clustered.csv",
                          str(args.min_pts), str(args.eps), str(args.threshold), str(args.frac)],
                         capture_output=True, text=True, check=True).stdout
    optics_ms = float(re.search(r"OPTICS ordering ([\d.]+) ms", out).group(1))
    k = int(re.search(r"significant clusters .*?: (\d+)", out).group(1))
    print(f"OPTICS  : found k={k} clusters automatically;  ordering = {optics_ms:8.0f} ms")

    # --- k-means (scikit-learn), told the same k ---------------------------
    # Warm up so timing reflects compute, not sklearn's one-time import/JIT/thread
    # pool start-up (the first .fit() in a process is much slower than the rest).
    KMeans(n_clusters=k, n_init=1, random_state=1).fit(pixels)

    def time_kmeans(n_init, reps):
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            KMeans(n_clusters=k, n_init=n_init, random_state=0).fit(pixels)
            best = min(best, (time.perf_counter() - t) * 1000.0)
        return math.ceil(best)  # round up to whole ms

    km1_ms = time_kmeans(1, reps=5)    # one k-means run (best of 5)
    km10_ms = time_kmeans(10, reps=2)  # the scikit-learn default (10 restarts)
    print(f"k-means : k={k} given;  fit (1 run) = {km1_ms:8.0f} ms   (n_init=10) = {km10_ms:8.0f} ms")

    # --- DBSCAN (scikit-learn), the closest relative of the OPTICS threshold cut --
    # A single global eps (here the same threshold OPTICS was cut at); finds k itself
    # and labels low-density pixels as noise, like OPTICS but without the ordering.
    DBSCAN(eps=args.threshold, min_samples=args.min_pts).fit(pixels[:1000])  # warm up
    t = time.perf_counter()
    db = DBSCAN(eps=args.threshold, min_samples=args.min_pts).fit(pixels)
    db_ms = math.ceil((time.perf_counter() - t) * 1000.0)
    db_k = len(set(db.labels_.tolist()) - {-1})
    db_noise = int((db.labels_ < 0).sum())
    print(f"DBSCAN  : eps={args.threshold} given;  found k={db_k}, {db_noise} noise px;  fit = {db_ms:8.0f} ms")

    print(f"\nratio OPTICS / k-means(n_init=1)  = {optics_ms / km1_ms:.2f}x")
    print(f"ratio OPTICS / k-means(n_init=10) = {optics_ms / km10_ms:.2f}x")
    print(f"ratio OPTICS / DBSCAN             = {optics_ms / db_ms:.2f}x")
    print("\nk-means is fastest but needs k and ignores noise/shape; DBSCAN finds k and noise "
          "at one global eps; OPTICS finds structure across densities and exposes the hierarchy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
