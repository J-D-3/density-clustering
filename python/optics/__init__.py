"""optics — OPTICS / HDBSCAN* density clustering for color spaces and N-D point clouds.

High-level (OpenCV image color pipelines)::

    import cv2, optics
    res = optics.cluster_image(cv2.imread("img.png"))   # hdbscan, Lab, voxel auto
    res.palette        # dominant colors (mean RGB + size), largest first
    res.labels         # per-pixel cluster ids (HxW), -1 = noise
    res.recolored()    # image painted with cluster mean colors

Low-level (any (N, Dim) cloud, Dim in 1..4) is re-exported from the native `_optics` module:
`hdbscan`, `shdbscan`, `cluster_threshold`, `extract_xi`, `soptics`, `compute_reachability`,
`quantize`. See their docstrings; for color, prefer Lab coordinates and metric='l2'.
"""

from __future__ import annotations

from . import spaces
from ._optics import (
    cluster_dbscan,
    cluster_threshold,
    compute_reachability,
    extract_xi,
    hdbscan,
    quantize,
    shdbscan,
    soptics,
)
from .color import Cluster, ImageClustering, cluster_image
from .spaces import lab_to_srgb, srgb_to_lab

__version__ = "0.9.1"

__all__ = [
    "cluster_image", "ImageClustering", "Cluster",
    "srgb_to_lab", "lab_to_srgb", "spaces",
    "hdbscan", "shdbscan", "cluster_threshold", "extract_xi", "soptics",
    "compute_reachability", "quantize", "cluster_dbscan",
    "__version__",
]
