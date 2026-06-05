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
    --max-dim 360 --min-pts 25 --eps -1 --threshold 30 --frac 0.01
```

Each cluster's pixels are drawn in the cluster's **mean color**, wrapped in a
transparent (approx. minimal) **enclosing sphere** of the same color, over a gray
backdrop (so near-white clusters stay visible). The tool also prints a per-cluster
pixel count + mean color. `colors_in.csv` / `colors_clustered.csv` are written in
the working dir.

### Output modes
- *(no `--out`/`--html`)* — opens an **interactive** matplotlib window: **drag to
  rotate, scroll to zoom**.
- `--out plot.png` — save a static image.
- `--html plot.html` — save an **interactive plotly** page (drag/zoom in a browser);
  needs `pip install plotly`.

Extra flags: `--true-color` colors each pixel by its own RGB instead of the cluster
mean; `--no-spheres` hides the enclosing spheres.

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

## Performance note (important for images)

OPTICS runtime is dominated by **neighborhood size**: each point is processed
against every other point within `eps`. Images often contain large **flat-color
regions** (a white or orange background), so with a generous `eps` a single
neighborhood can contain tens of thousands of points and the ordering pass
becomes ~O(region²) — seconds to minutes. Keep `eps` small (it shrinks
neighborhoods *and* separates color modes), and/or reduce `--max-dim`. The CSV
round-trip in this example is **not** the bottleneck (tens of ms); the algorithm
on dense data is.

`compare_kmeans.py` times the OPTICS ordering against scikit-learn k-means (told
the cluster count OPTICS found) on the same pixels:

```sh
python examples/color_clustering/compare_kmeans.py <image> \
    --exe build/examples/Release/optics_color.exe --max-dim 240 --eps 3 ...
```

Observed (these images, ~40–50k px): a single k-means fit is ~35–50 ms regardless
of parameters, while OPTICS ranges from ~0.3 s (small `eps`) to 10–25 s (large
`eps`, dense background). k-means is faster and runtime-stable but needs `k` and
force-assigns every pixel; OPTICS finds `k` itself and separates noise/edge
colors. Needs `pip install scikit-learn`.
