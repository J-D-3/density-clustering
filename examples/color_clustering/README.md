# Example: color-space clustering

Cluster the colors of an image with OPTICS and render the 3-D RGB color space.
This is the 3-D, ~1e6-point use case the library targets (color spaces), driven
end to end:

```
image  --(Python: Pillow)-->  RGB pixels (CSV)
       --(optics_color, C++ library)-->  per-pixel cluster labels (CSV) + summary
       --(Python: matplotlib)-->  3-D scatter of the clustered color space
```

The clustering itself is done by the **library** (`optics_color.cpp` calls
`optics::compute_reachability_dists` + `get_cluster_indices`); Python only does
image I/O and plotting.

## Build

`optics_color` is built with the project (when `OPTICS_BUILD_EXAMPLES` is ON):

```sh
cmake --preset msvc            # or linux-gcc / linux-clang
cmake --build --preset msvc
# -> build/examples/color_clustering/Release/optics_color(.exe)   (MSVC: Release subdir)
```

Python deps for the driver: `pillow`, `numpy`, `matplotlib`.

## Run

```sh
python examples/color_clustering/cluster_image.py <image> \
    --exe build/examples/Release/optics_color.exe \
    --max-dim 360 --min-pts 25 --eps -1 --threshold 30 --frac 0.01 \
    --out clusters.png
# add --rgb to color the scatter by each point's true color instead of by cluster
```

Outputs `clusters.png` (the 3-D plot) and prints a per-cluster pixel count + mean
color. `colors_in.csv` / `colors_clustered.csv` are written in the working dir.

## Parameters & tips

- `--max-dim` downscales (NEAREST, to preserve true pixel colors) so the longest
  side is at most this many pixels. Higher = more detail (and slower).
- `--eps` is the OPTICS generating distance in RGB units; `<= 0` auto-estimates.
  Smaller values keep tight color modes separate and let sparse "bridge" colors
  (anti-aliasing / JPEG gradients) fall out as noise.
- `--min-pts` / `--frac` control how dense / large a color must be to count as a
  cluster. A photo dominated by one color (e.g. a white background) will surface
  the dense colors first; lower these to also pick up small/sparse colors.
- `--threshold` is the reachability cut used to extract clusters from the ordering.

Because OPTICS is density-based, you do **not** specify the number of clusters —
it finds the color modes present, and reveals finer structure as you relax the
density parameters.
