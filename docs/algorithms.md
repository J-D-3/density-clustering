# The four clustering algorithms

This library ships **four** density-based clusterers: `OPTICS`, `sOPTICS`, `HDBSCAN*`, and
`sHDBSCAN`. They share machinery and a metric philosophy but answer different questions and scale
differently. This page explains how they relate, when to pick which, *why the statistical ("s")
variants use the cosine metric*, and *why those variants are slower on small data* — then lists the
efficiency work queued against them.

New here? Read the [README](../README.md) first for what OPTICS is and how to call it. This page is
the next layer down: the comparison and the design seams.

## TL;DR — which one do I want?

- **Low/medium dimension, exact result, don't know the data scale** → **OPTICS** (the default).
- **Low/medium dimension, want a parameter-light hierarchy that tolerates varying density** →
  **HDBSCAN\*** (only `min_cluster_size`; no `epsilon`).
- **High dimension (≳ 16-D) *or* very large `n` (≳ 1e5), and your similarity is angular/cosine** →
  the approximate **sOPTICS** / **sHDBSCAN**. They overtake the exact methods exactly in that regime.

The [1.0.0 benchmark matrix](benchmarking.md#100-reference-benchmark-matrix--results) (decisions
D1–D5) measured these crossovers; the defaults it confirmed are baked into this table.

## The map

All four ultimately produce density-based clusters, but they split along **two independent axes**:
*what structure they build* (a 1-D ordering vs. a cluster hierarchy) and *how they find neighbors*
(exact radius search vs. CEOs random projections).

| | **OPTICS** | **sOPTICS** | **HDBSCAN\*** | **sHDBSCAN** |
|---|---|---|---|---|
| Output | reachability ordering | reachability ordering | cluster hierarchy → flat labels | same |
| Neighbor source | exact KD-tree radius search | CEOs random projections | exact k-NN (same backend) | CEOs graph |
| Reachability | **directed** `max(core(o), d(o,p))` | directed (same driver) | **symmetric** `max(core(a),core(b),d(a,b))` | symmetric |
| Backbone | seed priority queue (`detail::optics_order`) | same driver | dense-Prim MST | Kruskal over sparse CEOs edges |
| Extraction | threshold cut or ξ steep-area | same | condense tree + EOM/leaf stability | same tail |
| Key params | `min_pts`, `epsilon` | `+ n_projections, k, m, seed` | `min_cluster_size`, `min_samples` | `+ projections, seed` |
| Metric | Euclidean (any backend metric) | **cosine** (L2/L1 via RFF) | Euclidean | **cosine** |
| Exact? | exact | approximate (Rand ≈ 1 vs exact) | exact | approximate |
| Complexity | ordering `O(n·|N_ε|)`; search-bound | `O(Dim·n·k·m)` fixed + ordering | **`O(n²)`** dense Prim | sparse MST + `O(n·D·Dim)` projection |
| Feasible to | search-bound, large `n` | high-d / large `n` | `n ≈ 1e4` (dense Prim) | `n ≥ 1e5` (only option there) |

Source: `compute_reachability_dists` / `compute_soptics_reachability_dists`
([`optics.hpp`](../include/optics/optics.hpp)) and `hdbscan` / `shdbscan`
([`hdbscan.hpp`](../include/optics/hdbscan.hpp)).

## The two design seams

Understanding the library is easier once you see that the four algorithms are really *two pairs*
crossed with *two interchangeable parts*.

### Seam 1 — OPTICS ↔ HDBSCAN\* differ in the graph, not the neighbor search

Both compute the **same** core distances from the **same** backend (`knn_core_dist`). The difference
is entirely downstream:

- **OPTICS** keeps reachability **directed** — `max(core(o), d(o,p))` — and walks it greedily with a
  seed priority queue into a 1-D *ordering* you then cut (threshold or ξ). You must supply `epsilon`.
- **HDBSCAN\*** **symmetrizes** it into *mutual reachability* — `max(core(a),core(b),d(a,b))` —
  builds a minimum spanning tree, and condenses the resulting hierarchy by cluster *stability*. No
  `epsilon`, and it naturally tolerates clusters at different densities. That is why HDBSCAN\* beats
  OPTICS on the mixed-density blobs where ξ extraction struggles
  ([benchmarking.md headline](benchmarking.md#headline)).

### Seam 2 — exact ↔ "s" differ in the neighbor search, not the extraction

The approximation lives **entirely in the neighbor graph**:

- `detail::optics_order` ([`optics.hpp`](../include/optics/optics.hpp)) is algorithm-agnostic: it
  takes a *neighbor provider* + *core-distance provider* as template closures. OPTICS feeds it a
  KD-tree; sOPTICS feeds it CEOs neighborhoods. Same ordering loop.
- HDBSCAN\*'s `detail::extract_from_mst` (single-linkage → condense → stability → EOM/leaf → labels)
  is **MST- and metric-agnostic by design**. `sHDBSCAN` swaps only the MST *source* (CEOs graph
  instead of dense Prim); everything after is shared.

Because the approximation is confined to the graph, the s-variants are validated by **statistical
agreement** (Rand/NMI ≈ 1.0 vs the exact method), not by bit-identical output.

## Why the statistical variants use the cosine metric

This is **forced by the CEOs math, not a preference.** CEOs (Concomitants of Extreme Order
Statistics; Xu & Pham, *Scalable DBSCAN with Random Projections*, NeurIPS 2024) finds neighbors like
this: project every point onto ~1024 random Gaussian vectors; for a query `q`, the few vectors on
which `q` has the most *extreme* (largest/smallest) projection are the directions most aligned with
`q`, and the points that are *also* extreme on those vectors are `q`'s approximate nearest neighbors
(see [`detail/random_projection.hpp`](../include/optics/detail/random_projection.hpp)).

This works because random projections **preserve inner-product order** with high probability — so the
method natively ranks neighbors by **dot product / angle**, *not* by Euclidean distance. It is
intrinsically an *angular* method.

To turn "ranks by inner product" into a usable distance, the library **L2-normalizes** every point
onto the unit sphere. On the sphere, Euclidean distance `d = √(2 − 2·cos θ)` is **strictly monotone
in cosine distance**, so the squared-Euclidean values CEOs already computes produce an ordering that
matches the cosine ordering exactly. **Cosine is the metric where CEOs is correct for free.**

**L2 / L1 are bolted on, not native.** Raw Euclidean/Manhattan don't preserve inner-product order, so
the only way to support them is to first embed points into **random Fourier features**
([`detail/random_features.hpp`](../include/optics/detail/random_features.hpp)) whose inner products
approximate a Gaussian (L2) or Laplacian (L1) kernel — both monotone-decreasing in the original
distance — then run the *same cosine pipeline* on the features. So L2/L1 cost an **extra embedding
pass** and are an approximation of an approximation. (χ² / Jensen-Shannon are not yet supported —
[#67](https://github.com/J-D-3/density-clustering/issues/67).)

> **Practical consequence:** scoring sOPTICS/sHDBSCAN on Euclidean toy layouts is a *metric mismatch,
> not a defect*. Evaluate them on direction-based ("cosine blob") data — that is why those datasets
> exist in [benchmarking.md](benchmarking.md#datasets).

## Why the statistical variants are slower on small datasets

The s-variants pay a **large fixed preprocessing cost** that is independent of how cheap the problem
actually is; the exact methods pay only for the work the data demands.

**The fixed tax.** Before any ordering/MST happens, CEOs must:

1. project all `n` points onto `D = 1024` Gaussian vectors → **`O(n·D·Dim)`** dot products
   (~16k ops/point at Dim = 16, regardless of how clustered the data is);
2. partial-sort each of the 1024 projection columns for per-vector extremes → **`O(D·n·log m)`**;
3. gather + ε-filter top-`k`×top-`m` candidates per point, then symmetrize.

For L2/L1, add another `O(n·Dim·FeatDim)` Fourier embedding on top.

**The exact path has almost no fixed cost.** Exact OPTICS builds one KD-tree, then does work
*proportional to neighborhood size*. On a small or sparse low-D cloud, neighborhoods are tiny and the
radius search is microseconds — there is nothing for the 1024-projection machinery to amortize
against. The crossover is roughly where the exact neighbor search becomes expensive, i.e. when
`n·D·Dim` is dwarfed by the exact search cost.

The mechanism is the **curse of dimensionality**: a static-`Dim` KD-tree degrades as `d` grows (exact
search visits ever more nodes), while CEOs query cost is essentially **dimension-independent**. So the
fixed projection overhead only pays off once dimension and/or `n` have made the exact search costly.
Below that, you are paying ~16M projection ops to dodge a radius search that would have taken
microseconds.

The matrix quantifies it ([benchmarking.md D2/D5](benchmarking.md#decisions-d1d5--every-one-confirms-the-shipped-defaults)):

- **sOPTICS** wins at `d ≥ 6 & n ≥ 1e5` (5.8–11.6×) and `d ≥ 32 & n = 1e4` (3.3–7.4×); exact wins at
  low d / small n.
- **sHDBSCAN** inherits the identical penalty (same CEOs front-end), so exact HDBSCAN\* stays the
  default up to `n ≈ 1e4`, and sHDBSCAN takes over at `n ≥ 1e5`, where dense-Prim's `O(n²)` is
  infeasible (14–43 s at `n = 1e5`).

> One-liner: **the exact methods are "pay for what the data costs"; the s-methods are "pay a flat tax
> up front, then ride a cheaper per-query rate."** Small data never recovers the tax.

## Efficiency roadmap

Where the headroom is, roughly by payoff. Tracked items link to their issue.

| | Idea | Targets | Status |
|---|------|---------|--------|
| **A** | **Exact sub-quadratic MST** for HDBSCAN\* — lifts the `n ≈ 1e4` dense-Prim wall without touching the cosine approximation. **Both landed:** `MstAlgorithm::Boruvka` (EXACT — same total MST weight as dense Prim — via Borůvka over a component-aware KD-tree; ~30× faster at n = 60k in low-D) and `MstAlgorithm::KnnGraph` (near-exact, Rand ≈ 1.0; the faster choice in high-D where KD-tree pruning degrades). | exact HDBSCAN\* scale | [**#66**](https://github.com/J-D-3/density-clustering/issues/66) · **1.0.0** |
| **C** | **Auto-dispatch front-end** — pick the engine from `n`/`d` so users never land on the wrong side of a crossover. **First slice landed:** `MstAlgorithm::Auto` selects the HDBSCAN\* MST backbone by (n, dim) from the [crossover sweep](#hdbscan-mst-backbone-auto-selection-72) below. **Remaining:** the OPTICS-side D1–D5 routing (exact-vs-s, Precompute-vs-OnDemand, structured projections). | usability / never-wrong-default | [**#72**](https://github.com/J-D-3/density-clustering/issues/72) · **1.0.0** |
| B | **Adaptive `D` / recall early-exit** for sOPTICS — scale `n_projections` with `n`/`d` and stop once recall stabilizes, shrinking the fixed tax and moving the crossover left. | sOPTICS small-n cost | backlog |
| D | **Auto-select structured (FHT) projections** past a dimension threshold (already opt-in, 1.2–1.4× at ≥ 64-D; folds into C). | sOPTICS high-d cost | backlog |
| E | **Share CEOs work** between sOPTICS and sHDBSCAN when both run on the same cloud. | redundant projection | backlog |
| F | **Parallelize the CEOs symmetrization** reverse-edge pass (currently single-threaded serial tail). | large-n sHDBSCAN | backlog |

The two highest-leverage items are **A** (removes a hard scale wall on the *exact* path) and **C**
(turns the benchmark study into a runtime policy). Both are queued for **1.0.0**; the rest are
opportunistic. The plan for A lives on [#66](https://github.com/J-D-3/density-clustering/issues/66).

## HDBSCAN\* MST-backbone auto-selection (#72)

`hdbscan(..., MstAlgorithm::Auto)` picks the MST backbone from `(n, dim)` instead of asking you to
know the crossover. The thresholds are **measured**, not guessed — from the `#66` crossover sweep
(`optics_hdbscan_mst_probe sweep`, 4 threads, well-separated blobs; `knn_rand` is KnnGraph's agreement
with exact Borůvka):

| dim | n = 8 000 | n = 32 000 | n = 64 000 | fastest |
|----:|----------:|-----------:|-----------:|---------|
| 3 | Bor 19 / Knn 31 | Bor 69 / Knn 141 | Bor 158 / Knn 288 | **Borůvka** (exact) |
| 8 | Bor 39 / Knn 50 | Bor 258 / Knn 308 | Bor 629 / Knn 792 | **Borůvka** (exact) |
| 12 | Bor 78 / Knn 59 | Bor 598 / Knn 504 | Bor 1755 / Knn 1756 | ~tie |
| 16 | Bor 119 / Knn 73 | Bor 725 / Knn 606 | Bor 2673 / Knn 2227 | **KnnGraph** (~exact) |
| 32 | Bor 271 / Knn 95 | Bor 2721 / Knn 943 | Bor 10963 / Knn 3854 | **KnnGraph** (2.8×) |

(ms; dense-Prim omitted — it is `O(n²)` and never the fastest, e.g. 5453 ms at n = 64k/3-D.)
`knn_rand = 1.0` across the grid (no measured quality loss on these blobs). The resulting policy
(`detail::resolve_auto_mst`):

- **n < 1 000** → `DensePrim` — every backbone is within a few ms; use the simplest, most-tested exact path.
- **dim < 16** → `Borůvka` — exact and fastest at low/mid dimension (wins ≤ 12-D, ties at 12 where
  exactness is the tie-breaker).
- **dim ≥ 16** → `KnnGraph` (near-exact, faster where the KD-tree prunes poorly) when the backend
  supports it, else `Borůvka` (still exact).

Auto is **opt-in** — the default stays `DensePrim` (exact, unchanged behaviour). Pass an explicit
backbone to override, e.g. `MstAlgorithm::Boruvka` to force exact in high dimension. The thresholds are
heuristics tuned for the multi-core default and the boundaries are soft; an explicit choice always wins.

## See also

- [README](../README.md) — what OPTICS is, the API, parameter choosing.
- [benchmarking.md](benchmarking.md) — the 1.0.0 reference matrix, decisions D1–D5, fairness rules.
- [perf/README.md](../perf/README.md) — scaling curves, per-backend and cross-library comparisons,
  the sOPTICS-vs-OPTICS crossover and projection-backend measurements.
