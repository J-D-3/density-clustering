# API stability & versioning policy (1.0.0)

This document is the **frozen public-API contract** for `density-clustering` from **1.0.0** onward.
It tells you, as a user of the library, exactly what you may depend on and what may change under you.

New here? The one-paragraph version: everything in the `optics::` namespace listed below is **stable** —
we will not break its signature or behaviour without a major-version bump. Everything in
`optics::detail::`, the vendored third-party code, and the test/benchmark helpers is **internal** and may
change in any release. When in doubt, depend only on what `#include <optics/optics.hpp>` and
`#include <optics/hdbscan.hpp>` give you.

## Semantic versioning

The project follows [Semantic Versioning 2.0.0](https://semver.org/). Given `MAJOR.MINOR.PATCH`:

- **MAJOR** — incremented for a breaking change to the stable surface below (a removed/renamed symbol, a
  changed signature or default argument, a changed struct layout, a removed enum value, or a changed
  meaning of an existing exact result). Upgrading across a major version may require code changes.
- **MINOR** — new backward-compatible API (new functions, new enum values appended at the end, new
  optional trailing parameters with defaults, new optional backend capabilities). Existing code keeps
  compiling and behaving. *Approximate*-algorithm outputs may also change here (see Tier 2).
- **PATCH** — backward-compatible bug fixes and performance work with no API change.

The C++ identity is part of the contract and will **not** change without a major bump: the `optics::`
namespace, the `<optics/optics.hpp>` header path, and the CMake package (`find_package(optics)`, target
`optics::optics`). The project *display* name changed from `OPTICS-Clustering` to `density-clustering`
before 1.0.0; the code identity above deliberately did not.

## What counts as the public surface

**Stable** = every symbol declared directly in `optics::` (and `optics::io::`) by the headers listed in
Tier 1. **Internal** = everything else: `optics::detail::`, the vendored `nanoflann.hpp` and `hnswlib/`,
and the utility namespaces `optics::testdata::` and `stopwatch::`.

A symbol's stability covers its **signature** (name, parameter types and order, default arguments,
return type), the **public layout** of any struct it returns or takes (field names, types, order), and
**enum values** (existing names keep their meaning; new values are only ever appended). For the *exact*
algorithms it additionally covers the **result** for a fixed input and version.

### Tier 1 — Stable (semver-protected)

| Header | Frozen public symbols |
|---|---|
| `optics.hpp` | `compute_reachability_dists`, `compute_soptics_reachability_dists`†, `cluster_threshold`, `extract_xi`, `get_cluster_indices`, `get_cluster_points`, `get_chi_clusters`, `get_chi_clusters_flat`, `epsilon_estimation`, `epsilon_estimation_knee`; `struct reachability_dist`; `enum class NeighborMode`, `enum class CoreDistMode`, `enum class Metric`†, `enum class SopticsProjection`†; aliases `Point`, `chi_cluster_indices`, `cluster_tree`. `cluster_dbscan` is **`[[deprecated]]`** (use `cluster_threshold`) and may be removed in a future major. |
| `hdbscan.hpp` | `hdbscan`, `shdbscan`†; `struct HdbscanResult`; `enum class ClusterSelectionMethod`, `enum class MstAlgorithm`. |
| `tree.hpp` | `class Node<T>`, `class Tree<T>` (the type behind `cluster_tree`), `tree_depth`, `tree_size`, `flatten_dfs`. |
| `preprocess.hpp` | `deduplicate`, `deduplicate_cosine`, `expand_clusters_to_original`, `quantize`; `struct DedupResult`. |
| `io.hpp` (`optics::io::`) | `cluster_labels`, `export_points_csv`, `export_reachability_csv`. |
| `backend.hpp` | The `NanoflannBackend<T,Dim,ApproxEpsPermille>` class and `ApproxNanoflannBackend` alias; the C++20 concepts that define the backend seam: `NeighborSearch` (required) and the optional capabilities `KnnCoreDist`, `RadiusSearchWithDists`, `KnnCoreDistWeighted`, `KnnGraph`. Backend authors program against these concepts. |
| `version.hpp` | `OPTICS_VERSION_MAJOR/MINOR/PATCH/STRING`, `OPTICS_VERSION`, `optics::version()`. |

**Optional backends** — `BoostRTreeBackend` (`boost_backend.hpp`, behind `OPTICS_ENABLE_BOOST_RTREE`)
and `HnswBackend` / `FastHnswBackend` (`hnsw_backend.hpp`, behind `OPTICS_ENABLE_HNSW`) are part of the
stable surface *when their build flag is on*. They are **off by default**, so they are not compiled in a
default build; their signatures still follow the policy above.

### Tier 2 — Stable signature, evolving results (the approximate algorithms)

Symbols marked **†** above belong to the randomized, approximate path: `compute_soptics_reachability_dists`,
`shdbscan`, and the `Metric` / `SopticsProjection` knobs that configure it.

Their **signatures and `seed`-determinism are frozen** like any Tier 1 symbol — same `seed` and inputs
on the same version reproduce exactly. But their **exact labels/orderings are not frozen across versions**:
a MINOR release may change them as the approximation (projections, recall, kernel features) improves.
They were always validated by Rand/NMI agreement with the exact algorithms, never bit-identical output,
so depend on *cluster quality*, not on specific label values. If you need a frozen result, use the exact
`compute_reachability_dists` / `hdbscan`.

### Tier 3 — Internal / not covered

These may change in **any** release, with no deprecation:

- **`optics::detail::`** — all algorithm internals (the ordering driver, MST backbones, random-projection
  and random-feature modules, the thread pool, profiling, math helpers). Do not include `detail/` headers
  or call `detail::` symbols from user code.
- **Vendored third-party** — `nanoflann.hpp` and `hnswlib/` are bundled implementation detail, wrapped by
  the backends above; use them through the backend seam, not directly.
- **Test/benchmark utilities** — `testdata.hpp` (`optics::testdata::` synthetic clouds) and `Stopwatch.hpp`
  (`stopwatch::` timing) exist for this repository's tests and benchmarks and carry no stability guarantee.

## Pre-1.0 behaviour changes folded into 1.0.0

The following deliberate behaviour changes were made *before* the freeze, on purpose, so that 1.0.0 ships
the right defaults and they never have to break again post-1.0. All are recorded in `CHANGELOG.md`:

- **Default neighbor acquisition is `OnDemand`** (since 0.9.1), not `Precompute` — O(one neighborhood)
  memory by default; `Precompute`/`Auto` remain opt-in.
- **Auto-epsilon defaults to the k-distance-knee estimator** (`epsilon_estimation_knee`, #57) instead of
  the uniform-density estimate, which over-shot on clustered data. Falls back to the uniform estimate for
  backends without `KnnCoreDist` and on degenerate inputs.
- **`cluster_threshold` and `extract_xi` deduplicate by default** (#46) — same partition on
  duplicate-free data; opt out with the trailing `dedup = false`.
- **sOPTICS auto-epsilon is data-scaled** (#58) rather than the old permissive `2.0`.

Pass an explicit `epsilon` / `mode` / `dedup` to recover any prior behaviour.

## For contributors

- Adding API: prefer a **new trailing parameter with a default**, a **new appended enum value**, or a
  **new function** — all MINOR-compatible. Never reorder parameters, change a default's meaning, or insert
  an enum value in the middle.
- Anything genuinely experimental goes in `optics::detail::` (or behind a build flag) until it is ready to
  be frozen, so the stable surface stays small and honest.
- Update this table and `CHANGELOG.md` in the same change that adds a stable symbol.
