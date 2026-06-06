#!/usr/bin/env bash
# One-command OPTICS demo (Linux / macOS).
#
# Builds the cluster_csv example, generates a sample dataset, clusters it, and
# renders a plot -- so a first-time user sees a result in a single command:
#
#   ./scripts/demo.sh
#
# Requires CMake + a C++20 compiler on PATH and the Python deps
# (pip install -r requirements.txt).

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

name="varied"
preset="${OPTICS_PRESET:-linux-gcc}"

echo "==> Configuring + building cluster_csv ($preset preset)"
cmake --preset "$preset"
cmake --build --preset "$preset" --target cluster_csv

exe="build/examples/cluster_csv"
[ -x "$exe" ] || { echo "cluster_csv not found at $exe" >&2; exit 1; }

echo "==> Generating sample dataset ($name)"
python3 tools/datasets.py --name "$name" --n 1500 --out "data/$name.csv"

echo "==> Clustering"
"$exe" "data/$name.csv" "data/$name" 10 -1 2.5 0.01

echo "==> Rendering plot -> data/${name}_plot.png"
python3 tools/visualize.py --points "data/${name}_points.csv" --reach "data/${name}_reach.csv" --out "data/${name}_plot.png"

echo
echo "Done. Open data/${name}_plot.png to see the clusters and the reachability plot."
