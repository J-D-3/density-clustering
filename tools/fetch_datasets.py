#!/usr/bin/env python3
"""Fetch Franti's clustering benchmark "shape sets" and convert them to this project's
CSV contract (a coordinates file ``x0,x1`` + a ground-truth ``label`` file).

These are widely-used third-party datasets with published clustering results -- the same
ones the FOPTICS paper reported on -- so scoring against them is comparable across papers
and tools. Source: P. Franti et al., University of Eastern Finland,
http://cs.uef.fi/sipu/datasets/ (please cite them if you publish results).

The files are not committed (data/ is gitignored); run this to (re)download them.

Usage:  python tools/fetch_datasets.py [out_dir=data/franti]
"""

import csv
import os
import sys
import urllib.request

BASE = "https://cs.uef.fi/sipu/datasets/"

# output-name : remote filename (the shape-set subset used by the FOPTICS paper, plus D31).
SETS = {
    "aggregation": "Aggregation.txt",
    "compound": "Compound.txt",
    "spiral": "spiral.txt",
    "R15": "R15.txt",
    "jain": "jain.txt",
    "flame": "flame.txt",
    "D31": "D31.txt",
}


def fetch_one(name, fname, out_dir):
    raw = urllib.request.urlopen(BASE + fname, timeout=30).read().decode("utf-8", "replace")
    pts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()  # files are whitespace- (tab/space-) separated
        if len(parts) < 3:
            continue
        pts.append((float(parts[0]), float(parts[1]), int(float(parts[2]))))

    coords_path = os.path.join(out_dir, f"{name}_coords.csv")
    truth_path = os.path.join(out_dir, f"{name}_truth.csv")
    with open(coords_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x0", "x1"])
        for x, y, _ in pts:
            w.writerow([f"{x:.6g}", f"{y:.6g}"])
    with open(truth_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"])
        for _, _, lab in pts:
            w.writerow([lab])
    return len(pts), len(set(p[2] for p in pts))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out_dir = argv[0] if argv else os.path.join("data", "franti")
    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    for name, fname in SETS.items():
        try:
            n, k = fetch_one(name, fname, out_dir)
            print(f"  {name:12s} {n:5d} points  {k:3d} labels  -> {out_dir}")
            ok += 1
        except Exception as e:  # noqa: BLE001  (report and continue)
            print(f"  {name:12s} FAILED: {e}")
    print(f"fetched {ok}/{len(SETS)} datasets into {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
