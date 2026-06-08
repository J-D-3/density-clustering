# optics_py — Python binding

An optional [pybind11](https://pybind11.readthedocs.io/) binding exposing OPTICS for
**1/2/3/4-D** NumPy point clouds. It is **off by default** and is the only part of the
project with a Python build dependency — the C++ library stays dependency-free.

## Build

```sh
pip install pybind11 numpy
cmake -S . -B build-py -DOPTICS_BUILD_PYTHON=ON \
      -DOPTICS_BUILD_TESTS=OFF -DOPTICS_BUILD_EXAMPLES=OFF \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build-py --config Release
```

This produces `optics_py.*.pyd` / `.so` under `build-py/python/` (a `Release/`
subdirectory on MSVC). Put that directory on `sys.path` or `PYTHONPATH` to import it.

## Use

```python
import numpy as np, optics_py
pts = np.random.default_rng(0).normal(size=(1000, 2))   # (N, Dim), Dim in 1..4

# Flat reachability-threshold cut -> per-point labels (-1 = noise).
# (The paper's ExtractDBSCAN, not a DBSCAN run. Omit threshold for an educated
#  default; cluster_dbscan is a deprecated alias of cluster_threshold.)
labels = optics_py.cluster_threshold(pts, min_pts=10, threshold=2.0)

# Hierarchical xi extraction -> per-point labels (same format as above)
labels_xi = optics_py.extract_xi(pts, min_pts=10, chi=0.05)

# Raw ordering: point_index + reachability arrays (in cluster order; -1 = UNDEFINED)
order = optics_py.compute_reachability(pts, min_pts=10)

# HDBSCAN* -> a separate density clusterer, no epsilon/threshold: just min_cluster_size
# (+ optional min_samples). Returns a dict: 'labels' (-1 = noise), 'probabilities' ([0,1]
# membership strength), 'n_clusters'. method = 'eom' (default) or 'leaf'.
res = optics_py.hdbscan(pts, min_cluster_size=15)
labels_h, probs, k = res["labels"], res["probabilities"], res["n_clusters"]
```

`hdbscan` deduplicates bit-identical points by default (`dedup=True`) — the big win on
flat-color/quantized data. The exposed functions cover plain OPTICS + HDBSCAN\*; sOPTICS
and sHDBSCAN live in the C++ headers but are not bound yet.

Smoke test: `python python/test_optics_py.py build-py/python/Release`.
