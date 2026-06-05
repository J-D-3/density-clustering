[![CI](https://github.com/J-D-3/OPTICS-Clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/J-D-3/OPTICS-Clustering/actions/workflows/ci.yml)

# OPTICS-Clustering

**Ordering Points To Identify the Clustering Structure ([OPTICS](https://github.com/J-D-3/OPTICS-Clustering/blob/master/background/OPTICS.pdf))** is an algorithm for finding density-based clusters in spatial data, presented by Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel and Jörg Sander in 1999.

## Introduction

This repository is a header-only **C++20** implementation of OPTICS. It provides an easy-to-use clustering algorithm that does not require knowing the number of clusters a priori, and that scales to large point clouds (millions of points). You can inspect the cluster structure visually (via a [reachability plot](https://github.com/J-D-3/OPTICS-Clustering/blob/master/resources/reachabilityplot.png)) and extract clusters either with a simple reachability threshold or with the hierarchical ξ (Xi) method.

For background on the algorithm see the [paper](https://github.com/J-D-3/OPTICS-Clustering/blob/master/background/OPTICS.pdf), [Wikipedia](https://en.wikipedia.org/wiki/OPTICS_algorithm), or [YouTube](https://www.youtube.com/watch?v=8kJjgowewOs).

## Usage

A point is a `std::array<T, Dim>` of Cartesian coordinates (`T` is `float` or `double`); a cloud is a `std::vector` of those.

```cpp
#include <optics/optics.hpp>
#include <array>
#include <vector>

using Point = std::array<double, 2>;

int main() {
    std::vector<Point> points = /* your data */;

    // The OPTICS cluster-ordering + reachability distances.
    auto reach = optics::compute_reachability_dists(points, /*min_pts=*/10);

    // Flat clusters by a reachability threshold ...
    auto clusters = optics::get_cluster_indices(reach, /*threshold=*/2.0);

    // ... or the hierarchical Xi method (nested cluster trees).
    auto trees = optics::get_chi_clusters(reach, /*chi=*/0.05, /*min_pts=*/10);
}
```

The full signature lets you choose the backend, neighbor-acquisition strategy, and thread count:

```cpp
optics::compute_reachability_dists<T, Dim, Backend>(
    points, min_pts,
    epsilon  = -1.0,                        // auto-estimated when <= 0
    mode     = optics::NeighborMode::Precompute,  // or OnDemand (lean memory)
    n_threads = 0);                         // 0 => hardware concurrency
```

### Convenience helpers

- `optics::cluster_dbscan(points, min_pts, threshold)` — compute the ordering and cut at a threshold in one call (returns one index list per cluster).
- `optics::extract_xi(reach_dists, chi, min_pts)` — Xi (steep-area) clusters as point-index lists.
- `optics::convert_cloud<float>(int_points)` — convert an integer/byte cloud (e.g. `uint8` color data) to a floating-point cloud, since `T` must be `float`/`double`.

### Visualizing results

The core writes no images; export CSV and render with the bundled script (matplotlib):

```cpp
#include <optics/io.hpp>
auto labels = optics::io::cluster_labels(points.size(), clusters);
optics::io::export_points_csv("points.csv", points, labels);   // any dimension
optics::io::export_reachability_csv("reach.csv", reach);
```

```sh
python tools/visualize.py --points points.csv --reach reach.csv --out plot.png
```

`visualize.py` handles 2D and 3D scatter (e.g. color spaces) and falls back to a PCA projection for higher dimensions.

## Dependencies

None required: [nanoflann](https://github.com/jlblancoc/nanoflann) (BSD 2-Clause) is vendored under `include/optics/`, and everything else is the C++ standard library. **Boost** is an optional alternative neighbor-search backend, enabled with `-DOPTICS_ENABLE_BOOST_RTREE=ON`. Bundled third-party licenses are listed in [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md).

## Building & testing

Header-only — just add `include/` to your include path and `#include <optics/optics.hpp>`. To build the tests/benchmark (CMake ≥ 3.21, MSVC 2022 / GCC 10+ / Clang 13+):

```sh
cmake --preset linux-gcc      # or: linux-clang, msvc
cmake --build --preset linux-gcc
ctest --preset linux-gcc
```

## License

Distributed under the MIT Software License (X11 license). (See accompanying file LICENSE.) Bundled third-party components and their licenses are listed in [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md).
