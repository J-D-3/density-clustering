#!/usr/bin/env python3
"""Test the high-level image color API (`optics.cluster_image`).

Run after `pip install ./python`:  python python/tests/test_color.py
"""
import numpy as np

import optics


def _three_block_image(rng):
    """A 120x120 RGB image of three horizontal color bands + mild per-pixel noise."""
    blocks = {"red": (200, 35, 35), "green": (35, 160, 70), "blue": (45, 65, 185)}
    rows = []
    for rgb in blocks.values():
        band = np.array(rgb, dtype=float) + rng.normal(0, 4.0, size=(40, 120, 3))
        rows.append(band)
    img = np.clip(np.vstack(rows), 0, 255).astype(np.uint8)
    return img, list(blocks.values())


def _nearest(palette_rgb, target):
    return min(palette_rgb, key=lambda p: sum((a - b) ** 2 for a, b in zip(p, target)) ** 0.5)


def main():
    rng = np.random.default_rng(0)
    img_rgb, blocks = _three_block_image(rng)  # this is RGB, not BGR

    # Default path: hdbscan, Lab, voxel auto. Input is RGB so bgr=False.
    res = optics.cluster_image(img_rgb, bgr=False)
    assert res.labels.shape == img_rgb.shape[:2], res.labels.shape
    assert res.n_clusters == 3, f"expected 3 clusters, got {res.n_clusters}"
    assert len(res.palette) == 3
    assert res.palette[0].size >= res.palette[-1].size  # sorted largest first

    # Each expected block color should appear as a cluster mean within a small tolerance.
    palette_rgb = [c.rgb for c in res.palette]
    for target in blocks:
        near = _nearest(palette_rgb, target)
        dist = sum((a - b) ** 2 for a, b in zip(near, target)) ** 0.5
        assert dist < 30, f"no cluster near {target}; closest {near} (dist {dist:.1f})"

    # recolored() returns an image of the same (downscaled) shape.
    rec = res.recolored()
    assert rec.shape == (*res.labels.shape, 3) and rec.dtype == np.uint8

    # BGR flag: feeding the BGR version with bgr=True must give the same palette colors.
    img_bgr = img_rgb[..., ::-1].copy()
    res_bgr = optics.cluster_image(img_bgr, bgr=True)
    assert res_bgr.n_clusters == 3
    assert {c.rgb for c in res_bgr.palette} == {c.rgb for c in res.palette}, "BGR/RGB paths must agree"

    # Other algorithms run and find the three blocks too.
    for algo in ("optics-xi", "optics-threshold", "shdbscan", "soptics"):
        r = optics.cluster_image(img_rgb, bgr=False, algo=algo)
        assert r.n_clusters >= 3, f"{algo} found {r.n_clusters} clusters"

    # (N, 3) point-cloud input (no image shape) also works.
    pts = img_rgb.reshape(-1, 3)
    rp = optics.cluster_image(pts, bgr=False)
    assert rp.labels.ndim == 1 and rp.labels.shape[0] == pts.shape[0]

    # space='rgb' path and explicit voxel.
    rr = optics.cluster_image(img_rgb, bgr=False, space="rgb", voxel=4)
    assert rr.n_clusters == 3

    print(f"test_color OK: cluster_image -> {res.n_clusters} clusters, palette {palette_rgb}; "
          "BGR/RGB/point/rgb-space/algos all pass")


if __name__ == "__main__":
    main()
