// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "nanoflann.hpp"

namespace optics {

// How the per-point epsilon-neighborhoods are obtained during the ordering pass.
//   Precompute : query every point's neighbors up front, in parallel, and cache
//                them. Fastest, but memory grows with the total neighbor count.
//   OnDemand   : query a point's neighbors when it is processed. Lean memory,
//                sequential. Prefer for very large clouds (e.g. 1e7 points).
enum class NeighborMode { Precompute, OnDemand };

// A neighbor-search backend ingests the point cloud once at construction (it may
// convert to a native layout there) and answers radius queries without any
// per-query reformatting of the query point. Backends satisfy this concept.
template <class B, class T, std::size_t Dim>
concept NeighborSearch = requires(const B b, const std::array<T, Dim>& p, T r,
                                  std::vector<std::size_t>& out) {
    b.radius_search(p, r, out);
};

namespace detail {

// nanoflann dataset adaptor that reads the user's std::vector<std::array<T,Dim>>
// in place -- no copy, no conversion on the query path.
template <typename T, std::size_t Dim>
struct PointCloudAdaptor {
    const std::vector<std::array<T, Dim>>* pts;

    explicit PointCloudAdaptor(const std::vector<std::array<T, Dim>>& points) : pts(&points) {}

    inline std::size_t kdtree_get_point_count() const { return pts->size(); }
    inline T kdtree_get_pt(std::size_t idx, std::size_t dim) const { return (*pts)[idx][dim]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

}  // namespace detail

// Default backend: nanoflann single-index KD-tree (runtime-sized, fast, header-only).
template <typename T, std::size_t Dim>
class NanoflannBackend {
public:
    using Point = std::array<T, Dim>;

    explicit NanoflannBackend(const std::vector<Point>& points, std::size_t leaf_max_size = 16)
        : adaptor_(points),
          index_(static_cast<int>(Dim), adaptor_,
                 nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size)) {
        index_.buildIndex();
    }

    // Append the indices of all points within Euclidean distance r of p to out.
    void radius_search(const Point& p, T r, std::vector<std::size_t>& out) const {
        // nanoflann's L2 metric works in squared distances.
        const T radius_sq = static_cast<T>(r) * static_cast<T>(r);
        thread_local std::vector<nanoflann::ResultItem<std::size_t, T>> matches;
        matches.clear();
        nanoflann::SearchParameters params;
        params.sorted = false;
        index_.radiusSearch(p.data(), radius_sq, matches, params);
        out.reserve(out.size() + matches.size());
        for (const auto& m : matches) out.push_back(m.first);
    }

private:
    using Adaptor = detail::PointCloudAdaptor<T, Dim>;
    using Metric = nanoflann::L2_Simple_Adaptor<T, Adaptor>;
    using Index = nanoflann::KDTreeSingleIndexAdaptor<Metric, Adaptor, static_cast<int>(Dim), std::size_t>;

    Adaptor adaptor_;
    Index index_;
};

}  // namespace optics

#ifdef OPTICS_ENABLE_BOOST_RTREE
#include "boost_backend.hpp"
#endif
