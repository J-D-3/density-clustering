"""High-level image color clustering — the OpenCV-friendly entry point.

`cluster_image` takes an image (OpenCV-style ``HxWx3`` uint8 BGR, or any ``(N, 3)`` color cloud),
clusters its colors, and returns per-pixel cluster labels plus a **palette** (the mean color and
size of each cluster) — the thing color clustering is usually for: palette extraction, recoloring,
or color segmentation.

Defaults are the optima from the study (``study/color_clustering/REPORT.md``):

==============  =========================  =====================================================
knob            default                    why
==============  =========================  =====================================================
``algo``        ``"hdbscan"``              best color recall, no epsilon/threshold to tune
``space``       ``"lab"``                  perceptual; separates orange/red, brown/red
``voxel``       ``"auto"`` (4 RGB / 2 Lab) ~12x fewer points, ~10x faster, recall preserved
``metric``      ``"l2"`` (approx algos)    cosine clusters by hue and merges black/white/gray
dedup           always on (in the library) lossless; the big speedup on flat-color images
==============  =========================  =====================================================

Guidance from the study: **do not gray-world / white-balance** before clustering — it shifts vivid
colors toward neutral gray/brown and destroys the very color identity you are clustering.

Example (OpenCV)::

    import cv2, optics
    bgr = cv2.imread("box.png")                 # HxWx3 uint8, BGR
    res = optics.cluster_image(bgr)             # hdbscan, Lab, voxel auto
    for c in res.palette:
        print(c.size, c.rgb)                    # dominant colors, largest first
    cv2.imwrite("segmented.png", res.recolored()[..., ::-1])  # RGB->BGR for cv2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _optics
from .spaces import srgb_to_lab

_ALGOS = ("hdbscan", "optics-xi", "optics-threshold", "shdbscan", "soptics")
_SPACES = ("lab", "rgb")


@dataclass
class Cluster:
    """One color cluster: its id, pixel count, fraction of the image, and mean color."""
    id: int
    size: int
    fraction: float
    rgb: tuple          # mean color, sRGB 0..255 ints
    lab: tuple          # mean color, CIELAB

    def __repr__(self) -> str:
        return f"Cluster(id={self.id}, size={self.size}, frac={self.fraction:.3f}, rgb={self.rgb})"


@dataclass
class ImageClustering:
    """Result of `cluster_image`: per-pixel labels + the color palette."""
    labels: np.ndarray          # HxW (image input) or N (point input); int, -1 = noise
    palette: list               # list[Cluster], largest first
    n_clusters: int
    noise_fraction: float
    _rgb: np.ndarray            # (N, 3) uint8 sRGB of the (possibly downscaled) pixels
    _shape: tuple | None        # (H, W) for image input, else None

    def recolored(self, background=(0, 0, 0)) -> np.ndarray:
        """Paint every pixel its cluster's mean RGB (noise -> `background`). Returns uint8 sRGB,
        shaped like the (possibly downscaled) input image (HxWx3) or (N, 3)."""
        out = np.empty((len(self.labels.reshape(-1)), 3), dtype=np.uint8)
        out[:] = np.asarray(background, dtype=np.uint8)
        flat = self.labels.reshape(-1)
        for c in self.palette:
            out[flat == c.id] = c.rgb
        return out.reshape((*self._shape, 3)) if self._shape else out


def _as_rgb_pixels(image, bgr: bool, max_dim):
    """Normalize input to (rgb_uint8 (N,3), shape (H,W) or None). Handles HxWx3 / (N,3),
    uint8 / float, BGR->RGB, and optional NEAREST downscale (image input only)."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        img = arr
        if max_dim is not None and max(img.shape[:2]) > max_dim:
            h, w = img.shape[:2]
            s = max_dim / max(h, w)
            nh, nw = max(1, round(h * s)), max(1, round(w * s))
            yi = np.minimum((np.arange(nh) * (h / nh)).astype(int), h - 1)
            xi = np.minimum((np.arange(nw) * (w / nw)).astype(int), w - 1)
            img = img[yi][:, xi]
        shape = img.shape[:2]
        flat = img.reshape(-1, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        shape, flat = None, arr
    else:
        raise ValueError("image must be HxWx3 (an image) or (N, 3) (a color cloud)")

    flat = np.ascontiguousarray(flat)
    if np.issubdtype(flat.dtype, np.floating) and flat.size and flat.max() <= 1.0:
        flat = flat * 255.0            # accept [0,1] floats
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    if bgr:
        flat = flat[:, ::-1]           # OpenCV BGR -> RGB
    return flat, shape


def _significance_filter(labels: np.ndarray, n: int, min_cluster_frac: float):
    """Drop clusters smaller than min_cluster_frac*n to noise (-1); compact ids by size (largest=0)."""
    min_size = max(1, int(min_cluster_frac * n))
    out = np.full(n, -1, dtype=np.int64)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    keep = [(int(c), int(i)) for i, c in zip(ids, counts) if c >= min_size]
    keep.sort(reverse=True)            # largest first -> new id 0,1,2,...
    for new_id, (_, old_id) in enumerate(keep):
        out[labels == old_id] = new_id
    return out, len(keep)


def cluster_image(image, *, algo: str = "hdbscan", space: str = "lab", voxel="auto",
                  bgr: bool = True, min_cluster_size=None, min_pts=None,
                  min_cluster_frac: float = 0.003, max_dim=None, metric: str = "l2",
                  seed: int = 42, n_threads: int = 0) -> ImageClustering:
    """Cluster an image's colors and return labels + palette.

    Parameters
    ----------
    image : HxWx3 (uint8/float, BGR by default for OpenCV) or (N, 3) color cloud.
    algo : 'hdbscan' (default, recommended), 'optics-xi', 'optics-threshold', 'shdbscan', 'soptics'.
    space : 'lab' (default, perceptual) or 'rgb'.
    voxel : 'auto' (4 in RGB units / 2 in Lab — the study sweet spot), a number, or 0/None to disable.
    bgr : True if `image` is OpenCV BGR (default); set False for RGB input.
    min_cluster_size, min_pts : density knobs; default size-relative (None => ~0.002*N / ~0.0007*N).
    min_cluster_frac : clusters smaller than this fraction of pixels become noise.
    max_dim : optional NEAREST downscale so the longest side <= max_dim (image input only).
    metric : 'l2' (default, recommended for color), 'l1', or 'cosine' — only used by shdbscan/soptics.
    seed : RNG seed for the approximate algos (deterministic).

    Returns
    -------
    ImageClustering with .labels (HxW or N), .palette (largest first), .n_clusters,
    .noise_fraction, and .recolored().
    """
    if algo not in _ALGOS:
        raise ValueError(f"algo must be one of {_ALGOS}, got {algo!r}")
    if space not in _SPACES:
        raise ValueError(f"space must be one of {_SPACES}, got {space!r}")

    rgb_u8, shape = _as_rgb_pixels(image, bgr, max_dim)
    n = len(rgb_u8)
    rgb_f = rgb_u8.astype(np.float64)
    lab = srgb_to_lab(rgb_f)
    coords = np.ascontiguousarray(lab if space == "lab" else rgb_f)

    if voxel == "auto":
        vox = 4.0 if space == "rgb" else 2.0
    elif voxel in (None, 0, 0.0):
        vox = 0.0
    else:
        vox = float(voxel)

    mcs = int(min_cluster_size) if min_cluster_size is not None else max(15, round(0.002 * n))
    mp = int(min_pts) if min_pts is not None else max(8, round(0.0007 * n))

    if algo == "hdbscan":
        labels = _optics.hdbscan(coords, mcs, n_threads=n_threads, voxel=vox)["labels"]
    elif algo == "shdbscan":
        labels = _optics.shdbscan(coords, mcs, metric=metric, seed=seed, n_threads=n_threads, voxel=vox)["labels"]
    elif algo == "optics-xi":
        labels = _optics.extract_xi(coords, mp, chi=0.05, voxel=vox)
    elif algo == "optics-threshold":
        labels = _optics.cluster_threshold(coords, mp, voxel=vox)
    else:  # soptics
        labels = _optics.soptics(coords, mp, extract="xi", metric=metric, seed=seed,
                                 n_threads=n_threads, voxel=vox)

    labels = np.asarray(labels).reshape(-1)
    labels, k = _significance_filter(labels, n, min_cluster_frac)

    palette = []
    for cid in range(k):
        mask = labels == cid
        cnt = int(mask.sum())
        mean_rgb = tuple(int(round(v)) for v in rgb_f[mask].mean(axis=0))
        mean_lab = tuple(round(float(v), 1) for v in lab[mask].mean(axis=0))
        palette.append(Cluster(cid, cnt, cnt / n, mean_rgb, mean_lab))
    palette.sort(key=lambda c: -c.size)
    noise_fraction = float((labels < 0).mean())

    out_labels = labels.reshape(shape) if shape else labels
    return ImageClustering(out_labels, palette, k, noise_fraction, rgb_u8, shape)
