# Documentation — sDBSCAN / sOPTICS sources

Reference material for implementing **sOPTICS** (scalable, random-projection OPTICS) in this library — the work
tracked by GitHub issue [#50](https://github.com/J-D-3/density-clustering/issues/50). The full implementation
plan lives in the local plan file `bright-conjuring-wren.md`.

New here? Read in this order: the OPTICS paper (what the ordering *is*) → the FOPTICS paper (the random-projection
idea, 2013 ancestor) → the sDBSCAN/sOPTICS paper (the modern algorithm we actually implement).

## Primary source (the algorithm we implement)

- **Scalable DBSCAN with Random Projections** — HaoChuan Xu & Ninh Pham, *NeurIPS 2024*.
  - Local copy: [`sDBSCAN_sOPTICS_RandomProjections_NeurIPS2024.pdf`](sDBSCAN_sOPTICS_RandomProjections_NeurIPS2024.pdf)
    (arXiv full version 2402.15679).
  - arXiv: <https://arxiv.org/abs/2402.15679> · NeurIPS:
    <https://proceedings.neurips.cc/paper_files/paper/2024/hash/31421b112e5f7faf4fc577b74e45dab2-Abstract-Conference.html>
  - Reference implementation (C++): <https://github.com/NinhPham/sDbscan>
  - **License note:** the `NinhPham/sDbscan` repo carries **no license** (all rights reserved). We therefore
    implement the algorithm **from the paper** — algorithms are not copyrightable — and **vendor no code**. The
    library stays dependency-free (we do **not** pull the repo's Eigen / FFHT / Pybind11 dependencies).

### What to take from it
- **sOPTICS** is the OPTICS-ordering variant (a reachability ordering for visualization / Eps selection) — the
  one that fits this reachability-based library. **sDBSCAN** is the flat variant; we get it for free by running
  our existing threshold/Xi extraction on the sOPTICS ordering.
- **CEOs** (Concomitants of Extreme Order Statistics): project all points onto `D` (~1024) Gaussian random
  vectors; for each point, its *extreme* (largest/smallest) projection vectors identify approximate near
  neighbors, because random projections preserve inner-product order with high probability.
- The approximate **core-distance** is the `minPts`-th nearest among the CEOs candidate set — so this library's
  existing `detail::compute_core_dist` is reused unchanged.
- CEOs is **angular** ⇒ cosine / inner-product is the primary metric; L1 / L2 / χ² / Jensen-Shannon come via
  random kernel features (tracked separately as issue
  [#51](https://github.com/J-D-3/density-clustering/issues/51)).

## Background / lineage (already in `../background/`)

- **OPTICS: Ordering Points To Identify the Clustering Structure** — Ankerst, Breunig, Kriegel, Sander,
  *SIGMOD 1999*. → `../background/OPTICS.pdf`
- **Fast Parameterless Density-Based Clustering via Random Projections (FOPTICS / DeBaRa)** — Schneider &
  Vlachos, *CIKM 2013*. The 2013 ancestor sDBSCAN/sOPTICS evolves. → `../background/FOPTICS_RandomProjections.pdf`

## Supporting theory (online)

- **CEOs / cross-polytope LSH** — the extreme-order-statistics property underpinning the candidate search
  (cited in the sDBSCAN paper; see its references for the CEOs and cross-polytope-LSH lineage).
- **Gan & Tao (2017)** — DBSCAN ↔ unit-spherical emptiness checking; the Ω(n^4/3) conditional lower bound that
  motivates approximate density clustering.

## Comparison targets (for the 1.0.0 benchmark trackers)

Relevant to issues [#53 (CPU-speed comparisons)](https://github.com/J-D-3/density-clustering/issues/53) and
[#54 (quality harness)](https://github.com/J-D-3/density-clustering/issues/54):

- **ELKI** — OPTICSXi, FastOPTICS, HDBSCAN* (the FastOPTICS/sOPTICS parity reference). <https://elki-project.github.io/>
- **mhahsler/dbscan** (R, ANN kd-tree C++ core) — fastest mainstream OPTICS. <https://github.com/mhahsler/dbscan>
- **NinhPham/sDbscan** (C++) — the direct sOPTICS competitor. <https://github.com/NinhPham/sDbscan>
- **scikit-learn-contrib/hdbscan** + **TutteInstitute/fast_hdbscan** — HDBSCAN* references (issue
  [#52](https://github.com/J-D-3/density-clustering/issues/52)).
