# Real-world image color clustering with OPTICS, HDBSCAN\* and their approximate variants

A study on 28 real product / magazine-cover images (14 originals + 14 preprocessed),
clustering their colors with five algorithms from this library in two color spaces,
scored against an explicit expectation of the colors a human would name. It measures
**computation time** and **correctness** (recall of expected colors), and the effect
of the supplied **preprocessing** and of the library's **dedup / voxel** point reduction.

All numbers below are produced by `run_study.py` (392 runs, 392 ok, 0 failed, 13 min
total, longest side downscaled to 256 px). Raw data: [`out/results.csv`](out/results.csv),
[`out/runs.json`](out/runs.json); aggregates: [`out/summary.md`](out/summary.md); plots:
[`out/plots/`](out/plots/).

> **How to read "correctness".** Photos have no per-pixel ground truth, so correctness is
> *recall of expected color roles* — did the algorithm surface a cluster of each color a
> person would name — scored by tolerant perceptual hue/lightness predicates (not Δ-E to an
> idealized swatch, which misfires on muted print colors). Cross-checked by inter-algorithm
> agreement (ARI/NMI) and same-article drift.

## TL;DR

1. **HDBSCAN\* is the most reliable color extractor** (mean recall 0.79), and it needs
   **no `epsilon`/threshold tuning**. The approximate `shdbscan` matches its recall (0.77)
   but is **~25× slower at this scale** — the random-projection variants only pay off at far
   larger `n` than a downscaled image.
2. **OPTICS + Xi (ξ=0.05) is a strong, cheap second** (0.74) and the fastest accurate option;
   the plain **threshold cut under-segments** (it merges a whole orange→red gradient into one
   muddy cluster) and is unstable across near-identical images.
3. **The supplied preprocessing is a double-edged sword:** background-removal + resize + dedup
   make it **~5× faster** (collapse 3.7×→10.9×), but the **gray-world white-balance *lowers*
   color correctness** (recall 0.71→0.63) because it turns vivid orange/red/white into muddy
   gray/brown — on the ChatGPT covers it destroys the colors entirely (recall→0.00).
4. **CIELAB ≥ RGB** overall (+0.03 mean recall) and decisively so for the fine extractors
   (OPTICS-Xi ξ=0.01: RGB 0.29 → Lab 0.77).
5. **Dedup is free and essential; voxel is a real speed knob** with a sweet spot at bin 4 (RGB)
   / bin 2 (Lab): ~12× collapse, ~10× faster, recall preserved. bin ≥ 8 over-merges (k→1).

---

## 1. Test set & expectations

| Article | Orig | Prep | Character | Expected roles (original) |
|---|---|---|---|---|
| Amlodipin 5 mg | 3 | 3 | tiny pharma-box photo | black bg, white box, blue band, red marker |
| Amlodipin 10 mg | 2 | 2 | tiny pharma-box photo | black bg, white box, blue band, red marker |
| Spiegel ChatGPT | 3 | 3 | orange bg, red border, white masthead/face | black bg, red, orange, white |
| Spiegel Hoffnung | 2 | 2 | green→blue bg, yellow band, brown text | black bg, red, yellow, white, blue, brown |
| Spiegel Mutante | 2 | 2 | black interior, red border, olive shapes | black, red, green, white |
| Bild Köln | 2 | 2 | photographic newspaper, busy | black bg, white newsprint, red, yellow |

Preprocessed images drop the *photographic black-background* role (background removal deletes
the surround). The preprocessing also **deskews/rotates, resizes ×0.5, mean-shifts, and applies
gray-world** — the last of which shifts hue (Mutante's red border → brown, a cyan cast appears),
so mean colors are **not** comparable 1:1 across the original↔preprocessed boundary; we compare
*which roles are recovered*, not RGB identity. (One stray `_prepped.png` export artifact is excluded.)

## 2. Method (one paragraph)

C++ harness `examples/color_clustering/color_study.cpp` clusters a 3-column CSV (RGB **or** Lab)
through the library's one-call wrappers, dedup-aware (issue #46): read → optional `quantize`
(voxel) → `deduplicate` to unique colors + weights → weight-aware clustering on the small unique
cloud → expand labels per pixel → significance filter (drop clusters < 0.3 % of pixels). It emits
one JSON line of metrics + a per-pixel label CSV. The Python orchestrator (`run_study.py`,
`study_lib.py`) builds the matrix **images × {RGB, Lab} × 7 configs**, converts sRGB→Lab in pure
numpy, scores recall, and computes ARI/NMI and same-article drift. Configs: `optics-thr`
(threshold cut), `optics-xi05/xi01` (Xi, ξ=0.05/0.01), `hdbscan-sm/lg` (exact HDBSCAN\*),
`shdbscan`, `soptics` (approximate, L2, 512 projections). `min_pts`/`min_cluster_size` scale
with pixel count.

---

## 3. Findings vs hypotheses

### H2 — Recall: which algorithm recovers the expected colors  ✅ HDBSCAN wins

Mean recall by algorithm × space × image-kind (1.0 = every expected color found):

| config | rgb·orig | rgb·prep | lab·orig | lab·prep | **overall** |
|---|---|---|---|---|---|
| optics-thr | 0.55 | 0.63 | 0.46 | 0.67 | 0.58 |
| optics-xi05 | 0.83 | 0.63 | 0.85 | 0.66 | **0.74** |
| optics-xi01 | 0.29 | 0.60 | 0.77 | 0.61 | 0.57 |
| hdbscan-sm | 0.90 | 0.63 | 0.90 | 0.63 | 0.76 |
| hdbscan-lg | 0.89 | 0.69 | 0.90 | 0.69 | **0.79** |
| shdbscan | 0.92 | 0.63 | 0.92 | 0.63 | **0.77** |
| soptics | 0.38 | 0.56 | 0.30 | 0.58 | 0.46 |

**Ranking: hdbscan-lg ≈ shdbscan ≈ hdbscan-sm ≈ optics-xi05 ≫ optics-thr ≈ optics-xi01 ≫ soptics.**
HDBSCAN\* is the most reliable and needs no `epsilon`/threshold. The **threshold cut
under-segments** — e.g. on Spiegel ChatGPT it lumps the entire orange→red gradient into one
muddy brown cluster (k=3, recall 0.50) while hdbscan splits orange/red/black/white (k=5, recall
1.00); see [`out/plots/scatter_Spiegel_ChatGPT_01.png`](out/plots/scatter_Spiegel_ChatGPT_01.png).
**Caveat (noise):** HDBSCAN\*'s EOM leaves 30–80 % of pixels as noise — it *finds the color
modes* but does not label every pixel; the OPTICS threshold cut labels almost everything (≤ 6 %
noise) but at lower color purity.

### H4 — Color space: Lab ≥ RGB  ✅ (decisive for fine extractors)

Mean recall is +0.03 higher in Lab overall, but the effect is concentrated where it matters: the
fine **OPTICS-Xi ξ=0.01 jumps RGB 0.29 → Lab 0.77**, and Xi/threshold separate perceptually
adjacent colors (orange/red, brown/red) far better in Lab. HDBSCAN\* is already near-ceiling in
both spaces. **Recommendation: cluster in Lab**, especially with OPTICS extractors.

### H1 — Background cluster (original vs preprocessed)  ⚠️ PARTLY FALSIFIED

Dominant (largest) cluster per article, `hdbscan-lg`:

| article | kind | dominant color | mean top-frac | black-dominant runs |
|---|---|---|---|---|
| amlodipin_5mg | original | white | 0.32 | 2/6 |
| amlodipin_5mg | preprocessed | white | 0.14 | 0/6 |
| spiegel_chatgpt | original | red | 0.46 | 0/6 |
| spiegel_chatgpt | preprocessed | brown | 0.63 | 0/6 |
| spiegel_hoffnung | original | blue | 0.24 | 0/4 |
| spiegel_mutante | original | **black** | 0.25 | **4/4** |
| spiegel_mutante | preprocessed | **black** | 0.24 | 3/4 |
| bild_koeln | original | gray | 0.31 | 0/4 |

The expectation "the black background becomes *a* cluster" **holds** (every original yields a
black cluster). But "it dominates" is **false for most**: the cover fills the frame, so white /
orange / red dominate; black is only the *largest* cluster for **Spiegel Mutante** (whose cover
*interior* is black). Preprocessing roughly **halves** the dominant fraction on Amlodipin
(background removed → more even), confirming background removal worked.

### H3 — Preprocessing effect  ✅ faster, ❌ less correct

| metric | original | preprocessed |
|---|---|---|
| dedup collapse (×) | 3.7 | **10.9** |
| OPTICS/MST core (ms) | 2624 | **509** |
| n_clusters | 38.0 | 42.9 |
| recall | **0.705** | 0.631 |
| noise frac | 0.283 | 0.226 |

Preprocessing is a **double-edged sword**. The resize + mean-shift **flatten regions → ~3× more
dedup collapse → ~5× faster**, and slightly cleaner (less noise). But **gray-world lowers
correctness**: it desaturates and hue-shifts vivid print colors toward neutral. On Spiegel
ChatGPT it is catastrophic — the original's clean orange `RGB(171,91,63)` / red `(157,47,46)` /
white `(208,228,235)` become **54 muddy gray/brown blobs** `RGB(118,70,81)`, `(137,130,110)` and
recall drops to **0.00** for *every* algorithm. **Verdict: keep the dedup/background-removal/resize
parts of a preprocessing pipeline; drop gray-world for color-identity work.**

### H5 — Same-article drift (consistency across images of one article)  ✅ quantified

Across the images of one article (after gray-world note above, originals only for color identity):
**matched-cluster colors are stable even when counts wobble.** With `hdbscan-lg`, the mean Δ-E
between matched clusters of two same-article images is small — Spiegel covers **ΔE ≈ 2.8–5.0**
(RGB count-spread often **0**), Amlodipin photos **ΔE ≈ 2.9–3.7** (counts spread up to 11 because
the small red marker drifts in/out of the significance threshold). The **threshold cut is far less
stable** (ΔE up to 17.6, e.g. Bild/Hoffnung) — another reason to prefer HDBSCAN\* or Xi. So: the
*number* of clusters can differ by a few between shots of the same article (mostly small features
crossing the size threshold), but the *colors* you do recover agree to a few Δ-E units.

### H6 — Approximate vs exact fidelity, and the speed it buys  ⚠️ data-dependent, no speed win here

Mean agreement of approximate vs exact partition:

| pair | original | preprocessed |
|---|---|---|
| shdbscan vs hdbscan-sm (ARI / NMI) | 0.55 / 0.63 | 0.82 / 0.86 |
| soptics vs optics-thr (ARI / NMI) | 0.28 / 0.31 | 0.75 / 0.78 |

Approximate methods **agree well on clean, flat preprocessed covers** but **diverge on busy
photographic originals** (Bild, gradient covers). And crucially, **at this scale they are slower,
not faster**: median core time is hdbscan ≈ 0.2 s vs **shdbscan ≈ 4.6 s (up to 33 s)**; optics
≈ 0.07 s vs soptics ≈ 0.7 s. The random-projection variants are built for `n` ≥ 10⁵–10⁶ where the
exact dense-Prim MST (`O(n²)`) and KD-tree become the bottleneck; on a 256-px image (≤ ~50 k px,
≤ ~20 k unique) **exact is both more accurate and ~25× faster**. Use `shdbscan`/`soptics` only at
genuine scale.

### Timing summary (median, all images)

| config | median core ms | median wall s | max wall s |
|---|---|---|---|
| optics-thr / xi05 / xi01 | ~65 | 0.53 | 1.1 |
| hdbscan-sm | 173 | 0.60 | 2.2 |
| hdbscan-lg | 233 | 0.65 | 2.3 |
| soptics | 700 | 1.19 | 3.8 |
| shdbscan | 4613 | 5.03 | 32.9 |

### Dedup & voxel sub-study (weighted / unique points)  ✅

**Dedup (lossless, issue #46) is always on and free:** it collapses identical pixels into weighted
unique points (3–6× on photographic originals, 14–19× on flat preprocessed/pharma boxes) with
*identical* clustering — it is what makes OPTICS on these dense clouds run in tens of ms instead of
seconds. **Voxel (lossy `quantize`) is a real speed/quality knob** (Bild & Mutante, `hdbscan-sm`):

| bin (RGB) | collapse | core ms | recall (Mutante) | recall (Bild) |
|---|---|---|---|---|
| 0 (dedup only) | 3–4× | 480–830 | 1.00 | 0.75 |
| 4 | ~12× | 58–65 | 1.00 | 0.75 |
| 8 | 34–52× | 7–12 | 0.75 | **0.00** |
| 16 | 130–240× | 4 | 0.25 | 0.00 |

**Sweet spot: bin 4 (RGB) / bin 2 (Lab) — ~12× collapse, ~10× faster, recall preserved.** Beyond
that (bin ≥ 8) over-quantization merges distinct colors and HDBSCAN\* collapses to a single cluster
(recall→0); the threshold cut sometimes *appears* to recover at very coarse bins, but that is the
grid manufacturing artificial modes, not real structure. Recommended image pipeline:
**dedup (always) + voxel bin≈4 + downscale**, in Lab.

---

## 4. Per-article expectation verdict (originals, best algorithm)

| Article | Expected | Recovered (hdbscan, Lab) | Verdict |
|---|---|---|---|
| Amlodipin 5/10 mg | black, white, blue, red | all four (red marker occasionally below size threshold) | ✅ |
| Spiegel ChatGPT | black, red, orange, white | all four; **orange *is* separable from red** with HDBSCAN/Xi (threshold merges them) | ✅ |
| Spiegel Hoffnung | black, red, yellow, white, blue, brown | yellow/white/blue/red yes; **brown text vs red** the hardest (recall ≈ 0.8) | ◑ |
| Spiegel Mutante | black, red, green, white | all four; black correctly dominant (cover interior) | ✅ |
| Bild Köln | black, white, red, yellow | white/red/yellow yes; photographic clutter → many small clusters + noise, recall ≈ 0.75 | ◑ |

## 5. Recommendations

* **Default for color clustering: `hdbscan` in CIELAB** — best recall, no `epsilon`/threshold to
  tune. If you need every pixel labelled (not just the modes), use **OPTICS + Xi (ξ≈0.05) in Lab**.
* **Avoid the plain threshold cut** for vivid/gradient images (it under-segments and is unstable);
  avoid `soptics` for correctness.
* **Image pipeline:** always dedup; add **voxel bin≈4** and downscale for speed; **skip gray-world**
  (and any aggressive white-balance) — it destroys the color identity you are trying to cluster.
* **Reserve `shdbscan`/`soptics` for `n` ≳ 10⁵** unique colors; below that, exact is faster *and*
  more accurate.

## 6. Reproduce

```sh
cmake --build --preset msvc --target color_study
cd study/color_clustering
python run_study.py --max-dim 256        # full matrix -> out/results.csv, out/runs.json
python summarize.py                      # -> out/summary.md
python make_plots.py                     # -> out/plots/*.png
```

Harness flags and the manifest/expectations live in `color_study.cpp`, `run_study.py`,
`study_lib.py`. Copyrighted cover images stay in the user's private folder; only derived numeric
data and abstract color-space plots are kept here.
