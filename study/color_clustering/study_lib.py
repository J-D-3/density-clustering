"""Shared helpers for the real-world color-clustering study.

This module is the glue around the C++ `color_study` harness. It:

  * enumerates the image test set and pairs originals with their preprocessed
    versions by *article* (the filenames don't line up 1:1 -- see `article_of`);
  * loads each image, optionally downscales it (NEAREST, to keep true colors),
    and produces the per-pixel RGB array plus a CIELAB array (pure-numpy sRGB->Lab,
    D65) -- the two color spaces the study clusters in;
  * runs the harness on a CSV of those coordinates and parses its one-line JSON;
  * scores a clustering against an *expectation* (a hand-listed set of color
    "roles" we expect a human to see) and against other clusterings (ARI/NMI).

There is no per-pixel ground truth for these photos, so "correctness" here means
"did the algorithm surface the colors a person would name", measured as recall of
the expected roles within a CIE76 Delta-E tolerance. See REPORT.md for the caveats.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# --- paths -------------------------------------------------------------------

IMAGES_ROOT = Path(r"C:\Users\ingop\OneDrive\01_Dokumente\AXON_Firma\ColorClustering")
PREPROCESSED = IMAGES_ROOT / "Preprocessed"
HARNESS = Path(r"C:\Repositories\density-clustering\build\examples\Release\color_study.exe")
OUT = Path(__file__).resolve().parent / "out"


# --- sRGB <-> CIELAB (D65), pure numpy --------------------------------------

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (...,3) uint8 or float in [0,255] -> Lab float (...,3). D65, CIE standard."""
    a = np.asarray(rgb, dtype=np.float64) / 255.0
    # inverse companding (sRGB -> linear)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    # linear RGB -> XYZ (sRGB/D65 matrix)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    # normalize by D65 white
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    A = 500.0 * (f[..., 0] - f[..., 1])
    B = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], axis=-1)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """CIE76 Delta-E (Euclidean distance in Lab)."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# --- test set manifest -------------------------------------------------------

ARTICLES = ["amlodipin_5mg", "amlodipin_10mg", "bild_koeln",
            "spiegel_chatgpt", "spiegel_hoffnung", "spiegel_mutante"]


def article_of(name: str) -> str | None:
    """Map a filename to its article key, tolerating the 5mg/05, 10mg/10 naming split."""
    n = name.lower()
    if "amlodipin" in n:
        if "5mg" in n or "_05_" in n or "_05" in n.replace("_05_01", "_05"):
            return "amlodipin_5mg"
        if "10mg" in n or "_10_" in n:
            return "amlodipin_10mg"
        # fall back: a bare "05"/"10" token
        if "05" in n:
            return "amlodipin_5mg"
        if "10" in n:
            return "amlodipin_10mg"
        return None
    if "bild" in n and "koeln" in n:
        return "bild_koeln"
    if "chatgpt" in n:
        return "spiegel_chatgpt"
    if "hoffnung" in n:
        return "spiegel_hoffnung"
    if "mutante" in n:
        return "spiegel_mutante"
    return None


@dataclass
class ImageEntry:
    path: Path
    article: str
    kind: str  # "original" | "preprocessed"

    @property
    def label(self) -> str:
        return f"{self.path.stem}[{self.kind[:4]}]"


def build_manifest() -> list[ImageEntry]:
    """Scan both folders, classify by article, exclude the stray `_prepped.png`."""
    entries: list[ImageEntry] = []
    for p in sorted(IMAGES_ROOT.glob("*.jpg")):
        art = article_of(p.stem)
        if art:
            entries.append(ImageEntry(p, art, "original"))
    for p in sorted(PREPROCESSED.glob("*.png")):
        if p.stem == "_prepped":      # stray empty-basename export artifact -> skip
            continue
        art = article_of(p.stem)
        if art:
            entries.append(ImageEntry(p, art, "preprocessed"))
    return entries


# --- image -> coordinate clouds ---------------------------------------------

def load_pixels(path: Path, max_dim: int = 320) -> np.ndarray:
    """Load an image as an (N,3) uint8 RGB array, NEAREST-downscaled so the longest
    side is at most `max_dim` (preserves true pixel colors; no new blended colors)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        s = max_dim / longest
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8).reshape(-1, 3)


def write_csv(coords: np.ndarray, path: Path) -> None:
    """Write an (N,3) float cloud as x0,x1,x2 CSV for the harness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, coords, fmt="%.4f", delimiter=",", header="x0,x1,x2", comments="")


# --- harness invocation ------------------------------------------------------

@dataclass
class RunResult:
    metrics: dict          # parsed JSON from the harness
    labels: np.ndarray     # per-pixel int labels (-1 noise), aligned to the pixel array
    ok: bool = True
    error: str = ""


def run_harness(csv: Path, algo: str, *, labels_out: Path, reach_out: Path | None = None,
                timeout: float = 180.0, **flags) -> RunResult:
    """Invoke color_study; return parsed JSON metrics + per-pixel labels."""
    cmd = [str(HARNESS), str(csv), "--algo", algo, "--labels-out", str(labels_out)]
    if reach_out is not None:
        cmd += ["--reach-out", str(reach_out)]
    for k, v in flags.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult({}, np.empty(0, int), ok=False, error=f"timeout>{timeout}s")
    if proc.returncode != 0:
        return RunResult({}, np.empty(0, int), ok=False,
                         error=(proc.stderr.strip().splitlines() or ["nonzero exit"])[-1])
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        metrics = json.loads(line)
    except json.JSONDecodeError:
        return RunResult({}, np.empty(0, int), ok=False, error=f"bad json: {line[:80]}")
    labels = read_labels(labels_out)
    return RunResult(metrics, labels)


def read_labels(path: Path) -> np.ndarray:
    """Read the last column (cluster_id) of a harness labels CSV, in row order."""
    arr = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, -1].astype(int)


# --- per-cluster summary -----------------------------------------------------

def cluster_summary(rgb: np.ndarray, lab: np.ndarray, labels: np.ndarray) -> list[dict]:
    """For each non-noise cluster: pixel count, fraction, mean RGB, mean Lab. Sorted by size desc."""
    out = []
    n = len(labels)
    for c in sorted(set(labels.tolist())):
        if c < 0:
            continue
        mask = labels == c
        cnt = int(mask.sum())
        mean_lab = lab[mask].mean(axis=0).round(1).tolist()
        out.append({
            "id": int(c), "px": cnt, "frac": cnt / n,
            "rgb": rgb[mask].mean(axis=0).round(1).tolist(),
            "lab": mean_lab,
            "name": primary_color(mean_lab),
        })
    out.sort(key=lambda d: -d["px"])
    return out


# --- expectations (the correctness yardstick) -------------------------------

# Named color anchors (approximate sRGB), converted to Lab once below.
_ANCHORS_RGB = {
    "black":  (12, 12, 12),
    "white":  (238, 238, 238),
    "red":    (200, 35, 35),
    "blue":   (40, 60, 150),
    "orange": (222, 120, 45),
    "yellow": (220, 190, 45),
    "green":  (110, 135, 75),   # olive/army green of the Mutante shapes
    "brown":  (95, 45, 35),     # dark red-brown "STIMMT" text
    "skin":   (200, 150, 120),
}
ANCHORS_LAB = {k: srgb_to_lab(np.array(v)).tolist() for k, v in _ANCHORS_RGB.items()}


@dataclass
class Role:
    color: str
    dominant: bool = False     # expected to be a large area (e.g. a background)
    background: bool = False    # photographic background -> only on originals


# Expected color roles per article. `background=True` roles are the photographic black
# surround that preprocessing removes (so they're dropped from preprocessed recall).
EXPECTATIONS: dict[str, list[Role]] = {
    "amlodipin_5mg":  [Role("black", dominant=True, background=True), Role("white", dominant=True),
                       Role("blue"), Role("red")],
    "amlodipin_10mg": [Role("black", dominant=True, background=True), Role("white", dominant=True),
                       Role("blue"), Role("red")],
    "spiegel_chatgpt":  [Role("black", dominant=True, background=True), Role("red"),
                         Role("orange", dominant=True), Role("white")],
    "spiegel_hoffnung": [Role("black", dominant=True, background=True), Role("red"),
                         Role("yellow"), Role("white"), Role("blue"), Role("brown")],
    "spiegel_mutante":  [Role("black", dominant=True), Role("red"), Role("green"), Role("white")],
    "bild_koeln":       [Role("black", dominant=True, background=True), Role("white", dominant=True),
                         Role("red"), Role("yellow")],
}


def expected_roles(article: str, kind: str) -> list[Role]:
    """Roles to score against. On preprocessed images, drop the photographic background role
    (background removal deletes the black surround -- except Mutante's is internal, not flagged)."""
    roles = EXPECTATIONS[article]
    if kind == "preprocessed":
        roles = [r for r in roles if not r.background]
    return roles


def _lch(lab) -> tuple[float, float, float]:
    """Lab -> (L, chroma, hue-degrees). Hue is the standard perceptual color angle."""
    import math
    L, a, b = lab
    C = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360.0
    return L, C, h


# Tolerant perceptual predicates that name a color from its Lab value. Far more robust than
# Delta-E to an idealized anchor: print/photo colors are muted (low chroma) and shifted in
# lightness, but their HUE is stable. A "gray" (low chroma, mid L) is the signature of an
# under-segmented, muddy cluster -- it matches no named hue, so it can't satisfy a role.
def _is(name):
    def black(L, C, h):  return L < 20
    def white(L, C, h):  return L >= 78 and C < 15
    def gray(L, C, h):   return C < 10 and 20 <= L < 78
    def red(L, C, h):    return C >= 12 and (8 <= h < 45)
    def orange(L, C, h): return C >= 20 and 45 <= h < 72 and L >= 38
    def yellow(L, C, h): return C >= 18 and 72 <= h < 110
    def green(L, C, h):  return C >= 10 and 110 <= h < 200
    def blue(L, C, h):   return C >= 8 and 200 <= h < 300
    def purple(L, C, h): return C >= 10 and 300 <= h < 340
    def brown(L, C, h):  return (8 <= h < 70) and L < 48 and 8 <= C < 42
    def skin(L, C, h):   return (20 <= h < 60) and 55 <= L < 82 and 8 <= C < 33
    return {k: v for k, v in locals().items() if k != "name"}[name]


COLOR_NAMES = ["black", "white", "gray", "red", "orange", "yellow",
               "green", "blue", "purple", "brown", "skin"]
PRED = {n: _is(n) for n in COLOR_NAMES}


def color_classes(lab) -> list[str]:
    """All color names whose predicate the Lab value satisfies (may be >1, e.g. red & brown)."""
    L, C, h = _lch(lab)
    return [n for n in COLOR_NAMES if PRED[n](L, C, h)]


def primary_color(lab) -> str:
    """Single best name for reporting/plot labels (first matching, else 'gray')."""
    cs = color_classes(lab)
    return cs[0] if cs else "gray"


def score_recall(summary: list[dict], roles: list[Role]) -> dict:
    """Recall of the expected color roles, by tolerant hue/lightness predicates.

    A role is hit if some significant cluster satisfies that color's predicate. Clusters are
    assigned to at most one role (greedy, largest cluster first) so a single muddy cluster
    can't satisfy several roles. Reports the matched cluster's size + RGB for transparency.
    """
    clusters = list(summary)
    cand = []  # (px, role_idx, cluster_idx) where the cluster's predicate matches the role color
    for ri, r in enumerate(roles):
        fn = PRED.get(r.color)
        if fn is None:
            continue
        for ci, c in enumerate(clusters):
            if fn(*_lch(c["lab"])):
                cand.append((c["px"], ri, ci))
    cand.sort(reverse=True)   # prefer assigning the largest matching cluster
    role_match = {ri: None for ri in range(len(roles))}
    used_c: set[int] = set()
    for px, ri, ci in cand:
        if role_match[ri] is not None or ci in used_c:
            continue
        role_match[ri] = ci
        used_c.add(ci)
    matches = []
    for ri, r in enumerate(roles):
        ci = role_match[ri]
        matches.append({
            "color": r.color, "hit": ci is not None,
            "cluster_px": clusters[ci]["px"] if ci is not None else None,
            "cluster_rgb": [int(x) for x in clusters[ci]["rgb"]] if ci is not None else None,
        })
    hits = sum(1 for m in matches if m["hit"])
    return {"recall": hits / len(roles) if roles else 0.0,
            "hits": hits, "n_roles": len(roles), "roles": matches}


# --- agreement metrics (pure numpy; sklearn import hangs in this env) --------

def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    c = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(c, (ia, ib), 1)
    return c


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Adjusted Rand Index + arithmetic NMI between two per-pixel labelings.

    Implemented from the definitions (noise label -1 is treated as just another label,
    matching sklearn) because `import sklearn` hangs in this environment.
    """
    c = _contingency(a, b).astype(np.float64)
    n = c.sum()
    if n < 2:
        return {}
    a_sum, b_sum = c.sum(axis=1), c.sum(axis=0)
    comb2 = lambda x: x * (x - 1.0) / 2.0
    sum_c, sum_a, sum_b = comb2(c).sum(), comb2(a_sum).sum(), comb2(b_sum).sum()
    cn2 = comb2(n)
    expected = sum_a * sum_b / cn2 if cn2 > 0 else 0.0
    max_idx = 0.5 * (sum_a + sum_b)
    ari = 1.0 if max_idx == expected else (sum_c - expected) / (max_idx - expected)
    # arithmetic NMI
    pij, pa, pb = c / n, a_sum / n, b_sum / n
    nz = c > 0
    mi = (pij[nz] * np.log(pij[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ent = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    denom = 0.5 * (ent(pa) + ent(pb))
    nmi = 1.0 if denom <= 0 else mi / denom
    return {"ari": round(float(ari), 4), "nmi": round(float(min(max(nmi, 0.0), 1.0)), 4)}


def match_clusters_across(sumA: list[dict], sumB: list[dict]) -> dict:
    """Greedy nearest-Lab matching between two images' clusters (same article). Reports the
    mean matched Delta-E and the cluster-count difference -- the same-article drift metric."""
    pairs = sorted((delta_e(a["lab"], b["lab"]), i, j)
                   for i, a in enumerate(sumA) for j, b in enumerate(sumB))
    used_a, used_b, dEs = set(), set(), []
    for dE, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); dEs.append(dE)
    return {"n_a": len(sumA), "n_b": len(sumB),
            "count_diff": abs(len(sumA) - len(sumB)),
            "matched": len(dEs),
            "mean_matched_dE": round(float(np.mean(dEs)), 1) if dEs else None,
            "max_matched_dE": round(float(np.max(dEs)), 1) if dEs else None}


if __name__ == "__main__":
    # Quick self-check: print the manifest and anchor Lab values.
    for e in build_manifest():
        print(f"{e.article:18s} {e.kind:12s} {e.path.name}")
    print("\nanchors (Lab):")
    for k, v in ANCHORS_LAB.items():
        print(f"  {k:7s} {[round(x,1) for x in v]}")
