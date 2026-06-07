#!/usr/bin/env python3
"""Smoke test for the optics_py pybind11 binding.

Run after building with -DOPTICS_BUILD_PYTHON=ON; pass the directory containing the
built module (e.g. build/python/Release) so it can be imported:

  python python/test_optics_py.py build/python/Release
"""
import sys

import numpy as np

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import optics_py  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    # Three well-separated 2-D blobs.
    centers = np.array([[0.0, 0.0], [40.0, 5.0], [10.0, 35.0]])
    pts = np.vstack([c + rng.normal(0, 1.5, size=(200, 2)) for c in centers])

    labels = optics_py.cluster_threshold(pts, min_pts=8, threshold=5.0, min_cluster_frac=0.05)
    assert labels.shape == (len(pts),), labels.shape
    n_clusters = len(set(int(v) for v in labels) - {-1})
    assert n_clusters == 3, f"expected 3 clusters, got {n_clusters}"

    # threshold omitted -> educated default; deprecated cluster_dbscan alias still works.
    assert optics_py.cluster_threshold(pts, min_pts=8).shape == (len(pts),)
    assert optics_py.cluster_dbscan(pts, min_pts=8, threshold=5.0).shape == (len(pts),)

    reach = optics_py.compute_reachability(pts, min_pts=8)
    assert reach["point_index"].shape == (len(pts),)
    assert reach["reachability"].shape == (len(pts),)
    assert reach["reachability"][0] == -1.0  # first ordered point is UNDEFINED

    xi = optics_py.extract_xi(pts, min_pts=8, chi=0.05)
    assert xi.shape == (len(pts),)

    # 1/3/4-D smoke (dispatch coverage).
    for dim in (1, 3, 4):
        p = rng.normal(0, 1.0, size=(60, dim)) + rng.integers(0, 3, size=(60, 1)) * 30.0
        out = optics_py.cluster_threshold(p, min_pts=5, threshold=5.0)
        assert out.shape == (60,)

    print(f"optics_py OK: cluster_threshold -> {n_clusters} clusters; reachability + xi + 1/3/4-D dispatch pass")


if __name__ == "__main__":
    main()
