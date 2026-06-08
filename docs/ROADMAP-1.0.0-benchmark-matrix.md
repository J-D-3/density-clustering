# Plan — the 1.0.0 benchmark matrix (single reference study)

**Status:** design complete + **amended for HDBSCAN\*/sHDBSCAN (2026-06-08)**, infrastructure
*not yet built*, study *not yet executed*. Tracked by **issue #59**; execution steps in
[`ROADMAP-1.0.0-execution.md`](ROADMAP-1.0.0-execution.md) (Part B). All gated features are now merged
(sOPTICS #50, non-Euclidean metrics #51, HDBSCAN\*/sHDBSCAN #52, HNSW backend #47, the comparison +
quality harnesses #53/#54); ELKI / NinhPham-sDbscan competitor columns are deferred past 1.0.0 into
the Docker repro env (§8.4). Its two jobs:

1. **Pick the library's data-dependent defaults**, with evidence — the **five** decisions in §1.
2. Be the project's **single, citable performance reference** against other libraries on the same
   data, on the same machine, at the same parameters.

It then gates the 1.0.0 tag: run the matrix → choose defaults → implement them → re-run a
confirmation subset → publish the tables/figures into [`benchmarking.md`](benchmarking.md) and the
README. Read [`../perf/README.md`](../perf/README.md) and [`benchmarking.md`](benchmarking.md)
first — this plan *extends* their methodology and folds the existing harnesses (§7) into one grid.

---

## 1. The decisions this study must resolve

Each default below is currently a hand-tuned guess or a single-anecdote choice. The matrix turns
each into a measured rule. We **write the decision rule as a falsifiable threshold up front** so the
study can confirm or refute it, rather than fishing for a story afterwards.

| # | Decision | Axes that resolve it | Candidate rule to confirm/refute |
|---|----------|----------------------|----------------------------------|
| D1 | **Backend by dimensionality** (exact nanoflann vs eps-approx nanoflann vs Boost R\*-tree vs HNSW #47) | `d` × `backend` × `recall`, fixed moderate `n` | "exact ≤ D\*; approx/HNSW above D\*, where measured speedup > 1.3× at recall ≥ 0.99." Find D\*. |
| D2 | **sOPTICS vs exact OPTICS** | `n` × `d` × `density` × `metric` | "sOPTICS when `n ≥ N*(d, density)` **and** the metric is angular/cosine-appropriate; else exact." Find the crossover surface `N*`. |
| D3 | **Neighbor-acquisition backend & mode per case** (which NN backend + Precompute vs OnDemand) | `density` (avg neighbors) × `n` × `d` × `mode` | "OnDemand when `avg_nbrs·n·8B > mem_budget` **or** cloud is dense; Precompute when sparse and it fits." Confirm the v0.9.1 default flip across the grid; find the avg-nbrs crossover. |
| D4 | **Epsilon estimator** (uniform vs k-distance-knee) | `density` × `k` (cluster structure) × **quality** (needs ground truth) | "knee is the default (already shipped #57); identify any regime where uniform is better." Quantify the quality gap per regime. |
| D5 | **HDBSCAN\* vs sHDBSCAN** (exact dense-Prim MST vs CEOs random-projection MST, #52) | `n` × `d` × `density` × `metric` × **quality** | "sHDBSCAN when `n ≥ M*(d, density)` **and** the metric is angular/cosine-appropriate; else exact HDBSCAN\*." Find the crossover `M*` (mirrors D2 for OPTICS/sOPTICS) and the quality cost. |

Stretch outcome: if the rules are clean, expose a tiny `optics::recommend_config(n, d,
density_hint, metric)` helper (does not exist today — grep confirms) that returns the chosen
backend/mode/algorithm, so the defaults are *machine-checked against the study*, not just prose.

---

## 2. Axes and levels

The user-requested grid, made precise. Each axis lists its levels and **whether it is swept fully or
spot-checked** — full-factorial is impossible (§3), so most axes are pinned in most tiers.

| Axis | Levels | Sweep policy |
|------|--------|--------------|
| **n** (points) | `round(10^(e/2))` for `e ∈ {4..14}` → 100, 316, 1 000, 3 162, 10 000, 31 623, 100 000, 316 228, 1 000 000, 3 162 278, 10 000 000 (11 levels) | Full along the **scaling spine** (Tier A); pinned at 1e4 / 1e5 elsewhere. |
| **d** (dims) | 1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128 (11 levels) | Full along the **dimensionality spine** (Tier B); pinned at {3, 16} elsewhere (the library's two stated targets). |
| **k** (clusters) | sampled from `1 .. 3d`: `{1, 2, ⌈d/2⌉, d, 2d, 3d}` deduped & clamped | Full in the **quality tier** (Tier C); pinned at a mid value elsewhere. |
| **density regime** | `dense` \| `sparse` \| `mixed` (definition below) | Full in Tier C; `mixed` elsewhere. |
| **engine / config** | see §4 (≈15 engine-configs) | Per-tier subset (some engines are infeasible past a size — §3). |
| **eps method** | `uniform` (`epsilon_estimation`), `knee` (`epsilon_estimation_knee`), plus `explicit` for same-eps fairness | Full in Tier C (quality); fixed to `knee` in timing tiers, **passed identically to every exact engine**. |
| **metric** | `euclidean`, `cosine` | `cosine` only for the sOPTICS / high-D angular datasets; `euclidean` elsewhere. |
| **min_pts** | main sweep fixed at **16** (the harness convention); spot-check {5, 10, 16, 50} on one slice | Pinned. |
| **dtype** | `float`, `double` | Pinned at `double`; one `float` spot-check (≈identical timing, half memory). |
| **threads** | fixed at **4** (`OPTICS_BENCH_THREADS`, the harness default — reproducible cross-machine) | Pinned; one separate thread-scaling study {1,2,4,8,16,all} on two cells. |

### Density regime — a precise, dimension-stable definition
"Density" must mean the same thing at d=2 and d=128, so define it by the **separation ratio**
`ρ = inter-cluster-separation / within-cluster-σ`, and target an **average ε-neighborhood size**:

- **dense** — `ρ ≈ 2`, blobs overlap; auto-ε neighborhoods are large (target avg_nbrs ≳ 0.05·n). The
  O(n²) regime; stresses D3/mode and core-dist processing.
- **sparse** — `ρ ≈ 10`, well separated; small neighborhoods (target avg_nbrs ≈ min_pts..50). The
  Precompute-friendly regime.
- **mixed** — clusters of **varied σ and varied points-per-cluster** + a **noise fraction**
  (e.g. 10% uniform background); the realistic case and the one where eps estimators (D4) diverge.

Pin the generator so a given `(n, d, k, regime, seed)` is byte-reproducible, and so neighborhood
size is controlled by `ρ` rather than drifting with `d`.

### 2.1. Expectation per axis (the falsifiable hypotheses)

A benchmark only earns its keep if it can **come out the other way**. So before running, we write down
what we expect each axis to show and the concrete observable that would **establish or refute** it. A
"refuted" expectation is a *result*, not a failure — several of these are deliberately set up so that
refuting them changes a default. Pair each row with the comparability rules in §9 (an expectation is
only testable if the comparison is fair).

| Axis | Expected pattern (H₁) | Observable that establishes / refutes it | Measured by |
|------|----------------------|-------------------------------------------|-------------|
| **n** | Sparse/mixed: ordering time grows **~O(n log n)**; dense + auto-ε: degrades to **~O(n²)** because `avg_nbrs` grows ~linearly with n. OnDemand peak memory ~flat in n·(tree); Precompute grows **O(n·avg_nbrs)** and hits a wall. | Log-log time slope: ≈1 (±0.15) on sparse, ≈2 on dense → established. Dense slope < 1.5 **refutes** the O(n²) framing. A Precompute run completing where the buffer > RAM **refutes** the memory model. | Tier A (`scale`, `mode_compare`) |
| **d** | Exact KD-tree per-query time rises sharply with d (curse); a **crossover D\*** exists (≈8–20) above which eps-approx / HNSW beat exact at **recall ≥ 0.99**. eps-approx recall ≈ 1.0 in low-d (nothing to prune ⇒ no speedup). Boost R\*-tree degrades faster than nanoflann in high-d. At fixed ρ, `avg_nbrs` stays ~constant across d. | Find D\* on the per-query-time × d curve. **No** crossover up to d=128 **refutes** D1 (exact always wins). avg_nbrs drifting with d **refutes** the density definition (and invalidates cross-d comparison). | Tier B (`benchmark`, `approx_probe`) |
| **k** (clusters) | At fixed (n, d, ρ), **ordering time is ~insensitive to k** (neighborhood size is set by ρ, not k). Quality: knee-eps ARI ~flat across k; **uniform-eps ARI degrades as k rises** (more, tighter clusters ⇒ bigger over-estimate). | Time flat in k (within noise) → established; strong rise **refutes** insensitivity. uniform-ARI falling while knee-ARI holds → established; both flat **refutes** the k-sensitivity of the estimator. | Tier C (`quality_compare`) |
| **density** | dense ⇒ O(n²), **OnDemand faster and the only feasible mode at scale** (confirms the v0.9.1 default flip); sparse ⇒ **Precompute faster** (small parallel cache, near-linear); approx backend gives **no** speedup on dense low-d (neighborhood-bound) but helps high-d sparse (search-bound). | Per-regime mode winner + time/memory. **Precompute beating OnDemand on dense refutes the v0.9.1 default.** approx beating exact on dense low-d **refutes** the "neighborhood-bound, not search-bound" claim. | Tier A/B (`mode_compare`, `approx_probe`) |
| **engine** | ours-OPTICS ≈ dbscan-R at equal eps (within ~2×), both **100–1000× faster than sklearn-OPTICS**; ours sits in **sklearn-DBSCAN's time band** while computing the full ordering+hierarchy. KMeans is always fastest (no neighbor graph) but lower quality on non-spherical/varied-density. ours-HDBSCAN\* quality ≈ sklearn/ELKI HDBSCAN\*; ELKI FastOPTICS ≈ our sOPTICS. | Same-machine ratio tables. ours-OPTICS > 2× slower than dbscan-R at equal eps → **regression flagged**. sklearn-OPTICS within 10× of ours **refutes** the headline speedup. KMeans matching OPTICS quality on Franti **refutes** the "shape/variety advantage". | Tier C/E (`quality_compare`, `quality_benchmark`, `timing_compare`, `timing_images`) |
| **eps method** (D4) | **knee ≥ uniform** in quality on clustered data, with large gaps at high-k / mixed (cf. R15 0.95 vs 0.43); they **converge on ~uniform-density / single-blob** data. knee ⇒ smaller ε ⇒ smaller neighborhoods ⇒ faster + less memory on dense. | Per-dataset ΔARI = ARI(knee) − ARI(uniform). A regime where uniform wins is a **real finding** (records where to prefer uniform). knee never helping anywhere **refutes** the shipped #57 default. | Tier C (`quality_compare --eps uniform\|knee`) |
| **metric** | sOPTICS (cosine) matches exact-OPTICS ARI **only on angular/cosine datasets**; on Euclidean layouts it is honestly lower (metric mismatch, not a bug). The 16-d cosine quality dip (ARI ≈0.57 vs ~0.82) is **addressed by the #58 data-scaled auto-eps** — confirm it no longer reproduces. | ARI(sOPTICS, cosine) ≈ ARI(exact) within tolerance **and** ARI(sOPTICS, euclidean) lower → established. sOPTICS matching exact on **Euclidean** too **refutes** the metric-mismatch framing. 16-d dip *re-appearing* under auto-eps **refutes** the #58 fix. | Tier C/D (cosine-blobs vs toys; `soptics_compare`) |
| **HDBSCAN\* / sHDBSCAN** (D5) | exact HDBSCAN\* time grows **~O(n²)** (dense-Prim MST), so a crossover `M*` exists above which sHDBSCAN (CEOs MST) wins — mirroring sOPTICS-vs-OPTICS (D2). sHDBSCAN matches exact-HDBSCAN labelling (Rand) **only in the angular regime**. ours-HDBSCAN\* quality ≈ sklearn HDBSCAN (ARI 0.99–1.00 already measured on 13 sets). | Find `M*` on the time × n curve; HDBSCAN\* slope ≈ 2 on dense established, < 1.5 refutes the O(n²) framing. Rand(sHDBSCAN, exact) high on cosine **and** lower on Euclidean → established. ours-HDBSCAN diverging from sklearn beyond tie-break noise **refutes** the parity. | Tier A/D (`hdbscan_compare`, `hdbscan_benchmark.py`) |
| **min_pts** | Larger min_pts ⇒ larger core distances ⇒ smoother reachability, fewer/larger clusters, larger neighborhoods ⇒ **slower**; quality has a **broad plateau** (robust), so the fixed default (16) is safe. | Time monotonic ↑ in min_pts; ARI unimodal/plateau. A **sharp** quality peak **refutes** "robust default" ⇒ min_pts needs its own per-data guidance. | min_pts spot-check {5,10,16,50} |
| **dtype** | float vs double: **~identical time** (memory-bandwidth / tree-traversal bound, not FLOP-bound) and ~identical clustering; float **halves memory** (≈2× higher Precompute ceiling). | \|t(float) − t(double)\| within run-to-run noise → established. float markedly faster **refutes** "not FLOP-bound". Cluster labels differing beyond tie-break noise **refutes** numerical equivalence. | dtype spot-check |
| **threads** | Precompute query phase scales **sub-linearly, saturating ~4–8 threads** (bandwidth-bound; 4 often ≈ all-cores); the sequential ordering loop is an **Amdahl ceiling**, capping end-to-end speedup (e.g. < 3× at 16 threads on dense). OnDemand barely scales. | Speedup-vs-threads curve. ~Linear scaling to 16 **refutes** the Amdahl ceiling (good news, would change the threads default). 4 ≈ all-cores **confirms** the harness's 4-thread default. | Thread-scaling study (2 cells) |

These also seed the **decision rules in §1**: D1 is the `d` row's crossover D\*, D2 the `metric`×`n`×`density`
crossover, D3 the `density` row's mode winner, D4 the `eps method` row's ΔARI map, D5 the
HDBSCAN\*/sHDBSCAN row's crossover `M*`.

---

## 3. Why not full-factorial — the reduction strategy

The naive grid is `11(n) × 11(d) × ~6(k) × 3(density) × ~15(engines) × 2(eps) × reps` ≈ **hundreds of
thousands of runs**, and a large fraction are *infeasible*: exact OPTICS and scikit-learn OPTICS are
~O(n²) on dense data, so e.g. `n=1e7, d=128, sklearn-OPTICS` would take days *per cell*. A naive grid
would never finish and would silently look "complete." Instead:

### 3a. Tiered (fractional) design — sweep one axis at a time around realistic baselines
- **Tier A — scaling spine.** Pin `d ∈ {3, 16}`, `density=mixed`, `k=⌈d/2⌉`; sweep **n full** ×
  engines (including **exact HDBSCAN\***, whose O(n²) dense-Prim MST is part of the scaling story).
  Yields the O(n)/O(n²) time curves, peak-memory curves, and the Precompute memory wall.
  *(Extends `scale.cpp` + `mode_compare.cpp`; HDBSCAN via `hdbscan_compare.cpp`.)*
- **Tier B — dimensionality spine.** Pin `n ∈ {1e4, 1e5}`, `density=mixed`; sweep **d full** ×
  backends × (exact/approx/HNSW) with **recall** captured; include exact HDBSCAN\* timing.
  Yields D1 (backend crossover D\*) and the curse-of-dimensionality picture.
  *(Extends `benchmark.cpp` + `approx_probe.cpp`.)*
- **Tier C — cluster-structure & quality.** Pin `n ∈ {3e3, 3e4}`, `d ∈ {2,3,8,16}`; sweep **k ×
  density × eps-method** with **ground truth** → ARI/NMI/Rand. Yields D4 (eps) and the quality
  baseline vs every engine (OPTICS/sOPTICS/HDBSCAN\*/sHDBSCAN + sklearn/dbscan-R).
  *(Extends `quality_compare.cpp` + `quality_benchmark.py` + `hdbscan_benchmark.py`.)*
- **Tier D — crossover probes.** Targeted 2-D slices where a default flips: sOPTICS-vs-OPTICS over
  `(n × density)` at d∈{8,16,64} (D2); **sHDBSCAN-vs-HDBSCAN\*** over `(n × density)` at d∈{8,16,64}
  (D5, same shape as D2); approx-vs-exact over `(d × eps-permille)` (D1/D3). Dense sampling only near
  the suspected crossover. *(Extends `soptics_compare.cpp` + `approx_probe.cpp`; D5 via `hdbscan_compare.cpp`.)*
- **Tier E — real-world.** §6 datasets across all applicable engines (anchors the synthetic story to
  reality and is the public comparison table). *(Extends `timing_images.py` + Franti via `quality_benchmark.py`.)*

This is a **star/fractional-factorial**: each tier varies one or two axes through their full range
while holding the rest at a realistic baseline, instead of the full cross-product. To catch
interactions the spines miss cheaply, add a **Latin-hypercube fill** of ~100 random interior cells
(random `(n,d,k,density,engine)`), run at low reps, flagged as exploratory.

### 3b. Feasibility gating — predict, cap, and **log** every skip
Before each cell, estimate cost from a cheap pilot (small-n fit of the engine's complexity) and:
- skip cells whose predicted wall-time > per-cell budget (e.g. 10 min) or predicted peak memory >
  machine RAM; **record the skip with its reason** in the output (never silently truncate — the
  benchmarking.md rule). scikit-learn OPTICS is already skipped > 8k px in the current tables; this
  generalizes that.
- auto-reduce repetitions as n grows (5 reps at small n → 1 at 1e7).
- enforce a hard timeout per run; a timed-out run is a recorded data point ("> T s"), not a hang.

### 3c. Budget
After Tier scoping, estimate total cell count (target: low thousands, not 10⁵) and wall-clock on the
reference machine; record it here before running. The study must be **resumable** (checkpoint + skip
completed cells) so it can run over several days.

---

## 4. Engines / configurations compared

| Family | Config | Source | Notes |
|--------|--------|--------|-------|
| **ours — OPTICS** | nanoflann exact; nanoflann approx (eps‰ ∈ {100,500,1000}); Boost R\*-tree; **HNSW (#47)** | this lib | × mode {Precompute, OnDemand} on the D3 slice |
| **ours — sOPTICS** | CEOs RP, cosine; vary `n_projections`/`k`/`m` on one slice | this lib | angular metric only — see comparability |
| **ours — HDBSCAN\*** | #52 exact dense-Prim mutual-reach MST + condensed-tree extraction | this lib | O(n²) MST ⇒ on the scaling/dim spines; quality vs sklearn/ELKI HDBSCAN\* |
| **ours — sHDBSCAN** | #52 approximate CEOs random-projection MST, cosine; vary `n_projections`/`k`/`m` | this lib | angular metric only — see comparability; the D5 scalable alternative to exact HDBSCAN\* |
| **scikit-learn** | OPTICS (xi), DBSCAN, HDBSCAN, **KMeans** (speed-floor baseline) | `quality_benchmark.py` / `timing_images.py` | OPTICS ~O(n²); gate > ~1e4 |
| **mhahsler/dbscan (R)** | ANN kd-tree OPTICS + Xi | `run_dbscan_r.R` | exact-Euclidean, **same eps** as ours |
| **ELKI (Java)** | OPTICSXi, FastOPTICS, HDBSCAN\* | manual, same CSVs | FastOPTICS = the sOPTICS parity reference (#53) |
| **NinhPham/sDbscan (C++)** | sOPTICS competitor | build from source | direct sOPTICS competitor (#53) |

eps method (D4) multiplies only the **exact-Euclidean** engines (ours-OPTICS, dbscan-R) and is held
identical across them per cell. KMeans is included purely as the "no neighbor graph" speed floor (it
already appears in `timing_images.py`); its quality is not comparable (needs `k`, spherical only).

---

## 5. Metrics captured (one tidy row per `dataset × engine × config × rep`)

**Speed/resource:** ordering/total wall-time (ms, median of reps + IQR), throughput (pts/s), peak
RSS (MB), avg ε-neighborhood size, per-phase breakdown (build/precompute/core_dist/relax/loop via
`-DOPTICS_PROFILE`), and for approx backends **neighbor recall** vs exact.

**Quality (ground-truth cells only):** ARI, NMI, Rand, predicted cluster count vs true, noise
fraction. Extraction held like-for-like (Xi at the same `chi`/`xi`; threshold at the same percentile).

**Derived:** speedup vs scikit-learn OPTICS (the headline ratio), recall/speed Pareto front per `d`,
quality/speed Pareto per dataset.

**Provenance per run:** seed, exact eps used, machine (CPU, RAM), OS, compiler + flags, library
commit, thread count, timestamp, engine version. Without these the "single reference" is not citable.

---

## 6. Datasets

**Synthetic** — from an *extended* `testdata.hpp` generator (today it only does `gaussian_blobs` /
`uniform_noise` / `make_blobs` with **no `k`/density control and no emitted ground-truth labels** —
see §8): parametrized by `(n, d, k, density-regime, noise-frac, seed)`, emitting the coords CSV **and**
the truth-label CSV (the existing CSV contract in `tools/README.md`). Add non-spherical shapes
(two-moons/spiral lifted to d-D) so OPTICS's arbitrary-shape advantage is exercised, not just blobs.

**Real-world:**
- **Franti shape sets** (Aggregation, Compound, spiral, R15, jain, flame, D31; 2-D, ground truth) via
  `tools/fetch_datasets.py` — already wired, published cross-paper results (cite Franti et al.).
- **RGB images** (3-D color clouds) via `timing_images.py` — the dense-cloud / O(n²) stress and the
  library's headline use case; sweep the pixel budget (the existing 800 → 100k curve).
- **Cosine blobs** (8-D, 16-D, L2-normalized) via `quality_benchmark.py:make_cosine_blobs` — the
  *only* fair quality test for sOPTICS (angular metric).
- **Add high-D real data** to make the dimensionality spine credible beyond synthetics: MNIST /
  Fashion-MNIST (784-D, with labels — reduce to {16,32,64,128} via PCA), GloVe / text embeddings
  (cosine, the sOPTICS sweet spot), and a **16-D perspective-transform set** matching the library's
  own stated target use case. *(New fetch scripts; document licensing as in `THIRD-PARTY-LICENSES.md`.)*

---

## 7. Existing tests folded into the matrix (nothing thrown away)

The "all-over-the-place" perf tests become the **per-cell executors** of one grid:

| Existing | Role in the matrix | Change needed |
|----------|--------------------|---------------|
| `scale.cpp` (`optics_scale`) | Tier A scaling spine | extend beyond d=3 to the d-spine; emit peak memory + tidy CSV |
| `benchmark.cpp` (`optics_benchmark`) | Tier B dimensionality spine | drive d/backend from CLI instead of 4 hard-coded scenarios |
| `backend_compare.cpp` + `timing_compare.py` | D1 backend axis on CSVs | add HNSW row (#47); already CSV-driven & tidy |
| `mode_compare.cpp` (`optics_mode_compare`) | D3 Precompute-vs-OnDemand memory/time | already emits avg_nbrs + projected GB + both times — ideal |
| `approx_probe.cpp` (`optics_approx_probe`) | D1/D3 recall-vs-speed Pareto | extend to the full d-sweep (the "approx-backend sweep study" in `ROADMAP-post-0.9.1.md` §2) |
| `quality_compare.cpp` + `quality_benchmark.py` | Tier C quality + D4 eps decision | already does uniform/knee/explicit eps, ARI/NMI/Rand, dbscan-R, sklearn |
| `soptics_compare.cpp` (`optics_soptics_compare`) | D2 sOPTICS-vs-OPTICS crossover | sweep n×density rather than 3 fixed blob scenarios |
| `hdbscan_compare.cpp` (#52, merged) | Tier A/B HDBSCAN\* timing + D5 HDBSCAN\*-vs-sHDBSCAN crossover | sweep n×d×density; emit timing + labels (it already cross-checks vs sklearn at ARI 0.99–1.00) |
| `hdbscan_benchmark.py` (#52, merged) | Tier C HDBSCAN\*/sHDBSCAN quality vs sklearn HDBSCAN + ground truth | already scores ARI/NMI/Rand + an ours-vs-sklearn agreement column; add fast_hdbscan / scikit-learn-contrib/hdbscan refs |
| `timing_images.py` | Tier E real images | already produces the headline real-world table |
| `perf.cpp` (`optics_perf`) | **kept separate** — microbench / regression gate (#48) | not a matrix cell; feeds CI perf gate, not the study |

The shared CSV contract (`csv_points.hpp` / `optics::io`) is the seam that lets every engine — ours,
sklearn, R, ELKI, sDbscan — consume the *same* generated dataset, which is exactly what makes the
comparison fair.

---

## 8. Infrastructure to build before running

1. **Generator extension** (`testdata.hpp` + a small `gen_dataset` CLI): `(n,d,k,density,noise,seed)`
   → coords CSV + truth CSV, with the density definition in §2 and non-spherical shapes. *(The single
   biggest gap — the current generators can't produce the requested grid.)*
2. **Dimension dispatch.** `Dim` is a compile-time template parameter, so a runtime d-sweep over
   {1,2,3,4,6,8,12,16,32,64,128} needs explicit instantiation of each harness for those 11 dims
   (a `switch(dim)` over instantiated templates). Watch build time / binary size; a single
   `optics_matrix` exe with these instantiations is cleaner than 11 binaries.
3. **One orchestrator** (`tools/run_matrix.py`): expands the tier specs into cells, applies
   feasibility gating + timeouts (§3b), invokes the C++ harnesses and the third-party adapters,
   appends one tidy long-format CSV/Parquet (`dataset,engine,config,metric,value,provenance…`),
   **checkpoints and resumes**, records environment metadata.
4. **Third-party adapters & repro env.** sklearn (have), dbscan-R (have), ELKI jar runner (new),
   NinhPham/sDbscan build+run (new), hnswlib (with #47). Containerize (Docker) so the "single
   reference" is reproducible off the dev box — R-sandbox/Java/build fragility is the main
   operational risk (see benchmarking.md's R note).
5. **Analysis notebook** (`tools/analyze_matrix.py`): from the tidy table produce the D1–D4 decision
   tables, the Pareto fronts, the speedup-vs-sklearn figures, and the regime map; output feeds
   `benchmarking.md`, the README, and (stretch) the `recommend_config` thresholds.
6. **Correctness gate before timing.** On every small cell, validate each engine's clustering against
   the brute-force reference (ARI≈1 vs exact on clean blobs) so we never benchmark a broken config.

---

## 9. Comparability rules (the fairness contract)

Lifted from [`benchmarking.md`](benchmarking.md) §"Comparability" and extended — every one of these,
if ignored, produces a misleading number:

1. **Same generating distance (eps)** across exact engines, per cell — emit it and pass it to
   dbscan-R/ELKI. (The single biggest past error: 50× artifacts on D31 from mismatched eps.)
2. **Metric matches method** — sOPTICS/sDbscan are cosine; score them on angular datasets only.
3. **Like-for-like extraction** — same Xi `chi`/`xi`, same threshold percentile; ordering ≠ extraction.
4. **Tie-breaking differs** (ours low-index-first; ELKI/dbscan high-index-first) → compare clusters
   (ARI/NMI/Rand) and timing, **never raw orderings**.
5. **Exclude I/O & startup** from the timed region on every side; time only the ordering.
6. **Ratios, not absolutes** — only same-machine engine ratios are reproducible; record full hardware
   provenance; published numbers sanity-check the *setup*, not absolute ms.
7. **Fixed thread count** (4) reported, with a separate thread-scaling study; warmup + reps + median;
   compare back-to-back runs, not against a stale baseline (microbench noise can reach ~30%).
8. **Determinism** — fixed seeds; sOPTICS is deterministic in seed but **report variance across ≥3
   seeds** since its output is randomized.

---

## 10. Phasing & exit criteria

- **Phase 0 — prerequisites:** all 1.0.0 features merged (#46, #47, #50–#52, #54–#55, #58 — ✅ done;
  ELKI/sDbscan deferred), generator + orchestrator + adapters built (§8), correctness gate green.
  This is the **current blocker** — none of §8 exists yet.
- **Phase 1 — pilot:** run Tiers at small n to fit cost models, finalize feasibility caps, publish
  the cell count + wall-clock budget here.
- **Phase 2 — full run:** Tiers A–E + LHS fill, checkpointed, on the reference machine (+container).
- **Phase 3 — decide & implement:** resolve D1–D4 into concrete defaults (and, stretch,
  `recommend_config`); change the library defaults; **re-run the confirmation subset** to verify the
  new defaults beat the old on the grid.
- **Phase 4 — publish & tag:** fold tables/figures into `benchmarking.md` + README as the citable
  reference; then tag **1.0.0**.

**Exit criteria:** every D1–D4 rule is backed by data (or explicitly recorded as "no significant
difference"); no silently-skipped cells (all gated skips logged with reasons); the reference tables
are reproducible from `run_matrix.py` on a clean container.

## 11. Risks

- **Combinatorial explosion** → tiered/fractional design + LHS fill (§3a), not full grid.
- **O(n²) infeasibility** → feasibility gating, OnDemand, downsampling, logged skips (§3b).
- **Third-party env fragility** (R sandbox, Java, building sDbscan) → containerized repro env (§8.4).
- **Run-to-run noise** → reps + median/IQR, fixed threads, same machine, back-to-back (§9.7).
- **Template-instantiation build cost** for 11 dims → single `optics_matrix` exe, monitor binary size (§8.2).
- **Fairness disputes** → every caveat documented (§9); the study is only as credible as its provenance.
