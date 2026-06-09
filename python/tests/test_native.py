#!/usr/bin/env python3
"""Smoke test for the native `_optics` extension (low-level binding).

After `pip install ./python` just run `python python/tests/test_native.py`.
For an in-tree build (-DOPTICS_BUILD_PYTHON=ON) pass the directory holding the built
module so it can be imported directly:

  python python/tests/test_native.py build/python/Release
"""
import sys

import numpy as np

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
    import _optics  # type: ignore  # noqa: E402
else:
    from optics import _optics  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [40.0, 5.0], [10.0, 35.0]])
    pts = np.vstack([c + rng.normal(0, 1.5, size=(200, 2)) for c in centers])

    labels = _optics.cluster_threshold(pts, min_pts=8, threshold=5.0, min_cluster_frac=0.05)
    assert labels.shape == (len(pts),), labels.shape
    n_clusters = len(set(int(v) for v in labels) - {-1})
    assert n_clusters == 3, f"expected 3 clusters, got {n_clusters}"

    # threshold omitted -> educated default; deprecated cluster_dbscan alias still works.
    assert _optics.cluster_threshold(pts, min_pts=8).shape == (len(pts),)
    assert _optics.cluster_dbscan(pts, min_pts=8, threshold=5.0).shape == (len(pts),)

    reach = _optics.compute_reachability(pts, min_pts=8)
    assert reach["reachability"][0] == -1.0  # first ordered point is UNDEFINED

    # extract_xi now takes min_cluster_frac (consistency with the other label fns).
    xi = _optics.extract_xi(pts, min_pts=8, chi=0.05, min_cluster_frac=0.05)
    assert xi.shape == (len(pts),)

    # HDBSCAN* -> labels + probabilities.
    hres = _optics.hdbscan(pts, min_cluster_size=20)
    assert set(hres.keys()) == {"labels", "probabilities", "n_clusters"}, hres.keys()
    assert hres["n_clusters"] == 3, f"hdbscan expected 3 clusters, got {hres['n_clusters']}"
    probs = hres["probabilities"]
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert _optics.hdbscan(pts, min_cluster_size=20, method="leaf")["n_clusters"] >= 1

    # sHDBSCAN -> same dict shape; deterministic in seed.
    sh = _optics.shdbscan(pts, min_cluster_size=20, metric="l2")
    assert set(sh.keys()) == {"labels", "probabilities", "n_clusters"}
    assert np.array_equal(_optics.shdbscan(pts, min_cluster_size=20, metric="l2", seed=1)["labels"],
                          _optics.shdbscan(pts, min_cluster_size=20, metric="l2", seed=1)["labels"]), \
        "shdbscan must be deterministic in seed"

    # sOPTICS -> per-point labels via threshold or Xi; deterministic in seed.
    so = _optics.soptics(pts, min_pts=8, extract="xi", metric="l2")
    assert so.shape == (len(pts),)
    assert _optics.soptics(pts, min_pts=8, extract="threshold", metric="l2").shape == (len(pts),)
    assert np.array_equal(_optics.soptics(pts, min_pts=8, seed=1, metric="l2"),
                          _optics.soptics(pts, min_pts=8, seed=1, metric="l2")), \
        "soptics must be deterministic in seed"

    # --- new: quantize utility + voxel kwarg -----------------------------------
    # A cloud with many near-but-not-equal colors; quantize snaps them onto a grid so they collapse.
    colors = (rng.integers(0, 4, size=(600, 1)) * 60).astype(float) + rng.normal(0, 2.0, size=(600, 3))
    q = _optics.quantize(colors, bin=16.0)
    assert q.shape == colors.shape
    assert len(np.unique(q, axis=0)) < len(np.unique(np.round(colors), axis=0)), "quantize should merge colors"
    # voxel kwarg threads through every clusterer without error and keeps per-point alignment.
    for fn, kw in [(_optics.cluster_threshold, dict(min_pts=8)),
                   (_optics.extract_xi, dict(min_pts=8)),
                   (_optics.soptics, dict(min_pts=8, metric="l2"))]:
        out = fn(colors, voxel=16.0, **kw)
        assert out.shape == (len(colors),)
    assert _optics.hdbscan(colors, min_cluster_size=20, voxel=16.0)["labels"].shape == (len(colors),)
    assert _optics.shdbscan(colors, min_cluster_size=20, metric="l2", voxel=16.0)["labels"].shape == (len(colors),)

    # 1/3/4-D dispatch coverage.
    for dim in (1, 3, 4):
        p = rng.normal(0, 1.0, size=(60, dim)) + rng.integers(0, 3, size=(60, 1)) * 30.0
        assert _optics.cluster_threshold(p, min_pts=5, threshold=5.0).shape == (60,)

    print("test_native OK: threshold/xi/reachability/hdbscan/shdbscan/soptics + quantize/voxel "
          "+ 1/3/4-D dispatch pass")


if __name__ == "__main__":
    main()
