# One-command OPTICS demo (Windows / PowerShell).
#
# Builds the cluster_csv example, generates a sample dataset, clusters it, and
# renders a plot -- so a first-time user sees a result in a single command:
#
#   pwsh scripts/demo.ps1            # or: powershell -File scripts/demo.ps1
#
# Requires CMake + a C++20 compiler on PATH and the Python deps
# (pip install -r requirements.txt).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$name = "varied"
Write-Host "==> Configuring + building cluster_csv (MSVC preset)"
cmake --preset msvc
cmake --build --preset msvc --target cluster_csv

$exe = Join-Path $root "build\examples\Release\cluster_csv.exe"
if (-not (Test-Path $exe)) { throw "cluster_csv not found at $exe" }

Write-Host "==> Generating sample dataset ($name)"
python tools/datasets.py --name $name --n 1500 --out "data/$name.csv"

Write-Host "==> Clustering"
& $exe "data/$name.csv" "data/$name" 10 -1 2.5 0.01

Write-Host "==> Rendering plot -> data/${name}_plot.png"
python tools/visualize.py --points "data/${name}_points.csv" --reach "data/${name}_reach.csv" --out "data/${name}_plot.png"

Write-Host ""
Write-Host "Done. Open data/${name}_plot.png to see the clusters and the reachability plot."
