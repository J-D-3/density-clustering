#!/usr/bin/env python3
"""Parametrized dataset generator for the 1.0.0 benchmark matrix (issue #59).

This is the generator the matrix design calls "the single biggest gap"
(`docs/ROADMAP-1.0.0-benchmark-matrix.md` section 8.1). It produces a cloud of
``n`` points in ``d`` dimensions with ``k`` clusters at a controlled *density
regime*, plus a noise fraction, and writes the **same CSV contract** every engine
in the matrix reads (ours via `csv_points.hpp`, scikit-learn, dbscan-R, ELKI):

  * coordinates file -- header ``x0,x1,...`` then one row per point
  * truth file       -- header ``label`` then one int per point (-1 = noise)

Generating once to CSV and having every engine read that file is exactly what
makes the cross-engine comparison fair (the design's section 9 fairness contract).

Why a separate file from ``datasets.py``: that module holds the fixed 2-D figure
generators the examples/figures depend on; this one is the matrix's arbitrary
``(n, d, k, density, noise, shape, seed)`` generator. It reuses ``write_csv`` from
there so the on-disk format stays identical.

Density regime -- the dimension-stable definition from the design section 2. We fix
the within-cluster standard deviation at ``sigma = 1`` and control the *separation
ratio* ``rho = inter-cluster-separation / sigma`` directly, so "dense" means the
same thing at d=2 and d=128 (it does NOT drift with d):

  * dense  -- rho ~ 2   : blobs overlap; auto-eps neighborhoods are large (the O(n^2) regime).
  * sparse -- rho ~ 10  : well separated; small neighborhoods (the Precompute-friendly regime).
  * mixed  -- clusters of *varied* sigma and *varied* points-per-cluster + a noise
              fraction (the realistic case, and where the eps estimators diverge -- D4).

Usage:
  python tools/gen_dataset.py --n 3000 --d 16 --k 8 --density mixed --noise 0.1 \
                              --shape blobs --seed 42 --out data/matrix/cell_0001.csv

Everything is deterministic in ``--seed`` so a given
``(n, d, k, density, noise, shape, seed)`` is byte-reproducible.
"""

import argparse
import math
import os

import numpy as np

from datasets import write_csv  # reuse the shared CSV contract

DENSITY_RHO = {"dense": 2.0, "sparse": 10.0}  # mixed varies sigma instead (see below)
SHAPES = ("blobs", "moons", "spiral")


def _place_centers(k, d, sep, rng):
    """Return k cluster centers in d-D whose *minimum pairwise distance* == ``sep``.

    Sample Gaussian centers, then rescale the whole set by ``sep / current_min_dist``.
    Rescaling is dimension-agnostic and makes the separation exact regardless of k/d
    (so the density regime is comparable across the d-spine). k=1 -> a single origin
    center (separation is undefined and unused)."""
    if k <= 1:
        return np.zeros((1, d), dtype=float)
    centers = rng.normal(0.0, 1.0, size=(k, d))
    diffs = centers[:, None, :] - centers[None, :, :]
    dmat = np.sqrt((diffs ** 2).sum(axis=2))
    iu = np.triu_indices(k, k=1)
    cur_min = float(dmat[iu].min())
    if cur_min < 1e-9:  # degenerate draw -- spread them out on an axis instead
        centers = np.zeros((k, d), dtype=float)
        centers[:, 0] = np.arange(k, dtype=float)
        cur_min = 1.0
    return centers * (sep / cur_min)


def _split_counts(n, k, rng, varied):
    """Partition n points among k clusters: equal (varied=False) or Dirichlet-varied."""
    if k <= 0:
        return []
    if not varied:
        base = [n // k] * k
        for i in range(n - sum(base)):
            base[i % k] += 1
        return base
    # varied populations: Dirichlet weights, floored at a sensible minimum.
    w = rng.dirichlet(np.ones(k) * 2.0)
    counts = np.maximum((w * n).astype(int), 5)
    # fix rounding so they sum to n
    while counts.sum() > n and (counts > 5).any():
        counts[counts.argmax()] -= 1
    counts[counts.argmin()] += n - counts.sum()
    return [int(c) for c in counts]


def make_blobs_nd(n, d, k, density, noise_frac, seed):
    """k isotropic Gaussian blobs in d-D at the requested density regime, + noise."""
    rng = np.random.default_rng(seed)
    sigma = 1.0
    n_noise = int(round(n * noise_frac))
    n_signal = n - n_noise

    varied = density == "mixed"
    # mixed clusters get their own sigma in [0.6, 1.5]; dense/sparse use the unit sigma.
    sigmas = rng.uniform(0.6, 1.5, size=k) if varied else np.full(k, sigma)
    # separation: dense/sparse use a fixed rho on the unit sigma. mixed uses a moderate rho
    # but, because its sigmas vary, scales the separation by the *largest* sigma so even the
    # widest two clusters stay separable (sep / max_sigma stays ~ rho_mixed).
    rho = 5.0 if varied else DENSITY_RHO[density]
    sep = rho * (float(sigmas.max()) if varied else sigma)

    centers = _place_centers(k, d, sep, rng)
    counts = _split_counts(n_signal, k, rng, varied)

    parts, labels = [], []
    for i, c in enumerate(counts):
        parts.append(rng.normal(centers[i % len(centers)], sigmas[i % k], size=(c, d)))
        labels += [i] * c

    coords = np.vstack(parts) if parts else np.empty((0, d))
    return _append_noise(coords, labels, n_noise, d, rng)


def _random_rotation(d, rng):
    """A random d-D orthonormal rotation (QR of a Gaussian matrix) so the lifted shape
    is not axis-aligned -- otherwise dims 0-1 would carry all the structure and the rest
    pure noise, which a feature-axis-aware method could exploit unfairly."""
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))  # fix signs for a deterministic proper rotation


def make_shape_nd(n, d, density, noise_frac, seed, shape):
    """A non-spherical 2-D shape (moons / spiral) embedded into d-D, + noise.

    The shape is built in the plane, given a *small* off-plane thickness (a thin
    manifold, NOT isotropic d-D noise -- otherwise the extra dims swamp the 2-D
    structure and density methods can't recover it), then rotated by a random
    orthonormal matrix so it is not axis-aligned. These exercise OPTICS's
    arbitrary-shape advantage that blobs do not. k is fixed by the shape (2)."""
    rng = np.random.default_rng(seed)
    n_noise = int(round(n * noise_frac))
    n_signal = n - n_noise
    # In-plane jitter scales with the shape; off-plane thickness is a SMALL absolute value
    # (independent of d) so the manifold stays thin and recoverable in any dimension.
    inplane = {"dense": 0.09, "sparse": 0.04, "mixed": 0.06}[density]
    offplane = 0.10
    scale = {"dense": 6.0, "sparse": 12.0, "mixed": 9.0}[density]

    n1 = n_signal // 2
    n2 = n_signal - n1
    if shape == "moons":
        t1 = math.pi * rng.random(n1)
        t2 = math.pi * rng.random(n2)
        a = np.c_[np.cos(t1), np.sin(t1)]
        b = np.c_[1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5]
        xy = np.vstack([a, b])
    else:  # spiral: two interleaved Archimedean arms, offset by pi, from a min radius
        u1 = rng.random(n1)
        u2 = rng.random(n2)
        r1 = 0.25 + 0.75 * np.sqrt(u1)  # min radius keeps the arms apart near the centre
        r2 = 0.25 + 0.75 * np.sqrt(u2)
        th1 = r1 * 2.5 * math.pi
        th2 = r2 * 2.5 * math.pi
        a = np.c_[r1 * np.cos(th1), r1 * np.sin(th1)]
        b = np.c_[r2 * np.cos(th2 + math.pi), r2 * np.sin(th2 + math.pi)]
        xy = np.vstack([a, b])
    labels = [0] * n1 + [1] * n2

    xy = xy * scale + rng.normal(0.0, inplane * scale, xy.shape)
    if d <= 2:
        coords = xy[:, :d]
    else:
        rest = rng.normal(0.0, offplane, size=(n_signal, d - 2))
        coords = np.hstack([xy, rest]) @ _random_rotation(d, rng).T
    return _append_noise(coords, labels, n_noise, d, rng)


def _append_noise(coords, labels, n_noise, d, rng):
    """Append uniform background noise (label -1) over the data's bounding box, then
    shuffle so point order does not encode the labels."""
    labels = list(labels)
    if n_noise > 0 and len(coords) > 0:
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        noise = lo + rng.random((n_noise, d)) * span
        coords = np.vstack([coords, noise])
        labels += [-1] * n_noise
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels, dtype=int)
    perm = rng.permutation(len(coords))
    return coords[perm], labels[perm]


def generate(n, d, k, density, noise_frac, shape, seed):
    if density not in ("dense", "sparse", "mixed"):
        raise ValueError(f"density must be dense|sparse|mixed, got {density!r}")
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if shape == "blobs":
        return make_blobs_nd(n, d, k, density, noise_frac, seed)
    return make_shape_nd(n, d, density, noise_frac, seed, shape)


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate one benchmark-matrix dataset (CSV).")
    p.add_argument("--n", type=int, required=True, help="total number of points (incl. noise)")
    p.add_argument("--d", type=int, required=True, help="dimensionality")
    p.add_argument("--k", type=int, default=3, help="cluster count (blobs only; shapes fix k=2)")
    p.add_argument("--density", choices=("dense", "sparse", "mixed"), default="mixed")
    p.add_argument("--noise", type=float, default=0.0, help="noise fraction in [0,1)")
    p.add_argument("--shape", choices=SHAPES, default="blobs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="coordinates CSV path (truth -> <stem>_truth.csv)")
    args = p.parse_args(argv)

    coords, labels = generate(args.n, args.d, args.k, args.density, args.noise, args.shape, args.seed)
    coords_path, truth_path = write_csv(coords, labels, args.out)
    n_clusters = len(set(int(v) for v in labels) - {-1})
    n_noise = int((labels < 0).sum())
    print(f"wrote {coords_path} ({len(coords)} points, {args.d}-D, "
          f"density={args.density}, shape={args.shape})")
    print(f"wrote {truth_path} ({n_clusters} true clusters, {n_noise} noise points)")


if __name__ == "__main__":
    main()
