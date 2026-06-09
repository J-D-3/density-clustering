"""Color-space conversions used by the image API — pure NumPy, no OpenCV dependency.

The clustering itself is Euclidean, so the *space* you cluster in decides which colors are
considered "close". The study (``study/color_clustering``) found **CIELAB** best: its Euclidean
distance approximates perceived color difference, so it separates perceptually adjacent colors
(orange/red, brown/red) that RGB muddles.

CIELAB here is the CIE standard, **D65** white point, with ``L`` in [0, 100] and ``a``/``b`` roughly
[-128, 127] — *not* OpenCV's 8-bit Lab (which rescales ``L`` to [0, 255] and offsets ``a``/``b`` by
128). We use the float convention so distances match the study exactly; never feed ``cv2.cvtColor(...,
COLOR_BGR2LAB)`` output here.
"""

from __future__ import annotations

import numpy as np

# sRGB (D65) <-> linear-RGB <-> XYZ matrices (IEC 61966-2-1).
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB -> CIELAB (D65). `rgb`: (..., 3) uint8 or float in [0, 255]. Returns float Lab (..., 3)."""
    a = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _RGB2XYZ.T) / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    A = 500.0 * (f[..., 0] - f[..., 1])
    B = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (D65) -> sRGB uint8 (..., 3). Inverse of `srgb_to_lab`, for recoloring/visualization."""
    lab = np.asarray(lab, dtype=np.float64)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + A / 500.0
    fz = fy - B / 200.0
    fx3, fz3 = fx ** 3, fz ** 3
    xr = np.where(fx3 > _EPS, fx3, (116.0 * fx - 16.0) / _KAPPA)
    yr = np.where(L > _KAPPA * _EPS, fy ** 3, L / _KAPPA)
    zr = np.where(fz3 > _EPS, fz3, (116.0 * fz - 16.0) / _KAPPA)
    xyz = np.stack([xr, yr, zr], axis=-1) * _WHITE_D65
    lin = xyz @ _XYZ2RGB.T
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 0.0) ** (1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
