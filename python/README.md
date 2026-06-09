# optics — Python package

OPTICS / HDBSCAN\* density clustering for **color spaces** and N-D point clouds, with a
high-level, **OpenCV-friendly image color API**. Built on the header-only C++ library via a
[pybind11](https://pybind11.readthedocs.io/) extension; the C++ library itself stays
dependency-free.

The defaults encode what the color-clustering study (`study/color_clustering/REPORT.md`) found
works best on real images: **HDBSCAN\* in CIELAB, voxel-quantized, L2 metric, dedup always on.**

## Install

```sh
pip install ./python            # from the repo root (scikit-build-core builds the C++ extension)
```

Needs a C++20 compiler + CMake ≥ 3.21 (the build pulls `scikit-build-core` and `pybind11`
itself). Runtime dependency: `numpy`. OpenCV is **not** required — images are plain NumPy arrays;
`pip install ./python[opencv]` only if you want `cv2` in your own pipeline.

## Quickstart — cluster an image's colors (OpenCV)

```python
import cv2, optics

bgr = cv2.imread("box.png")                 # HxWx3 uint8, BGR (OpenCV's order)
res = optics.cluster_image(bgr)             # defaults: hdbscan, Lab, voxel auto, L2

print(res.n_clusters, "colors;", f"{res.noise_fraction:.0%} noise")
for c in res.palette:                       # dominant colors, largest first
    print(f"  {c.size:6d} px  rgb={c.rgb}")

labels = res.labels                         # HxW int array, -1 = noise
cv2.imwrite("segmented.png", res.recolored()[..., ::-1])   # mean-color image, RGB->BGR for cv2
```

`cluster_image` accepts an `HxWx3` image (uint8 or float, **BGR by default** — pass `bgr=False`
for RGB) or any `(N, 3)` color cloud. It returns an `ImageClustering` with `.labels`,
`.palette` (list of `Cluster(id, size, fraction, rgb, lab)`), `.n_clusters`, `.noise_fraction`,
and `.recolored(background=(0,0,0))`.

### Recommended defaults (from the study)

| knob | default | why |
|---|---|---|
| `algo` | `"hdbscan"` | best color recall, no `epsilon`/`threshold` to tune |
| `space` | `"lab"` | perceptual — separates orange/red, brown/red that RGB muddles |
| `voxel` | `"auto"` (4 in RGB units / 2 in Lab) | ~12× fewer points, ~10× faster, recall preserved; ≥ 8 over-merges |
| `metric` | `"l2"` | for the approximate algos; `"cosine"` clusters by hue and merges black/white/gray |
| dedup | always on (in the library) | lossless; the big speedup on flat-color images |

> **Don't gray-world / white-balance before clustering.** The study found it shifts vivid colors
> toward neutral gray/brown and destroys the color identity you are trying to recover.

### Tuning

```python
optics.cluster_image(bgr, algo="optics-xi")          # label every pixel (hdbscan leaves noise)
optics.cluster_image(bgr, space="rgb", voxel=4)       # cluster in RGB instead of Lab
optics.cluster_image(bgr, min_cluster_size=200)       # require larger color regions
optics.cluster_image(bgr, max_dim=512)                # downscale big images first (speed)
optics.cluster_image(bgr, algo="shdbscan")            # approximate; only worth it at >~1e5 colors
```

`algo` ∈ `{"hdbscan", "optics-xi", "optics-threshold", "shdbscan", "soptics"}`. HDBSCAN\* finds the
color *modes* but labels much of the image as noise (−1); the OPTICS extractors label almost every
pixel but at lower color purity. The approximate variants (`shdbscan`/`soptics`) are for very large
clouds — below ~10⁵ unique colors, exact `hdbscan` is both more accurate **and** faster.

## Low-level API

The native functions operate on a raw `(N, Dim)` cloud (`Dim` ∈ 1..4) and are re-exported on the
package: `optics.hdbscan`, `optics.shdbscan`, `optics.cluster_threshold`, `optics.extract_xi`,
`optics.soptics`, `optics.compute_reachability`, and `optics.quantize`. Every clusterer takes an
optional `voxel` (grid size; 0 = off). For color work pass Lab coordinates and `metric="l2"`.

```python
import numpy as np, optics
pts = np.random.default_rng(0).normal(size=(1000, 3)) * 30
res = optics.hdbscan(pts, min_cluster_size=15, voxel=4.0)   # dict: labels, probabilities, n_clusters
lab = optics.srgb_to_lab(rgb_u8)                            # color-space helpers also exported
```

## Build in-tree (no pip)

For development you can build just the extension from the top-level CMake:

```sh
cmake --preset msvc -DOPTICS_BUILD_PYTHON=ON \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build --preset msvc --target _optics
python python/tests/test_native.py build/python/Release   # pass the dir holding the built _optics
```

## Tests

```sh
python python/tests/test_native.py     # low-level binding (after pip install)
python python/tests/test_color.py      # high-level cluster_image
```
