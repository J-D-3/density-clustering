// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

// nanoflann is vendored third-party; silence its warnings (e.g. MSVC C4324/C4127)
// so the project can build warnings-as-errors without modifying the vendored header.
#if defined(_MSC_VER)
#  pragma warning(push, 0)
#endif
#include "nanoflann.hpp"
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

namespace optics {

// How the per-point epsilon-neighborhoods are obtained during the ordering pass.
//   Precompute : query every point's neighbors up front, in parallel, and cache
//                them. Fastest, but memory grows with the total neighbor count.
//   OnDemand   : query a point's neighbors when it is processed. Lean memory,
//                sequential. Prefer for very large clouds (e.g. 1e7 points).
enum class NeighborMode { Precompute, OnDemand };

// How a point's core-distance is computed during the ordering pass.
//   Scan : nth_element over the point's full epsilon-neighborhood (default;
//          exact, and cheap when neighborhoods are small).
//   Knn  : a k-NN query for the min_pts-th nearest neighbor, avoiding a scan of
//          huge neighborhoods -- a large win on dense clouds (e.g. flat-color
//          images, issue #24). Yields identical core-distances (hence identical
//          orderings) to Scan. Requires the backend to model KnnCoreDist; for
//          backends without that capability the ordering loop falls back to Scan.
enum class CoreDistMode { Scan, Knn };

// A neighbor-search backend ingests the point cloud once at construction (it may
// convert to a native layout there) and answers radius queries without any
// per-query reformatting of the query point. Backends satisfy this concept.
template <class B, class T, std::size_t Dim>
concept NeighborSearch = requires(const B b, const std::array<T, Dim>& p, T r,
                                  std::vector<std::size_t>& out) {
    b.radius_search(p, r, out);
};

// Optional capability: a backend that can answer the core-distance directly via a
// k-NN query (distance to the min_pts-th nearest point, or nullopt if it lies
// beyond r). Enables CoreDistMode::Knn; detected with `if constexpr`.
template <class B, class T, std::size_t Dim>
concept KnnCoreDist = requires(const B b, const std::array<T, Dim>& p, std::size_t k, T r) {
    b.knn_core_dist(p, k, r);
};

// Optional capability: a backend that returns each neighbor's SQUARED distance alongside
// its index, reusing the distances the search already computed. Lets the ordering loop
// skip recomputing those distances in the core-distance scan and the relaxation (issue
// #55). Enabled only for double coordinates, where the backend's accumulated squared
// distance is bit-identical to detail::square_dist (so the ordering is unchanged);
// detected with `if constexpr`.
template <class B, class T, std::size_t Dim>
concept RadiusSearchWithDists = std::is_same_v<T, double> &&
    requires(const B b, const std::array<T, Dim>& p, T r,
             std::vector<std::size_t>& out, std::vector<double>& out_sq) {
    b.radius_search_with_dists(p, r, out, out_sq);
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
//
// ApproxEpsPermille selects nanoflann's eps-approximate search: a query prunes any
// branch whose minimum distance exceeds worstDist / (1 + eps), visiting fewer nodes
// at the cost of possibly missing points near the query boundary. 0 (the default)
// is exact and bit-for-bit unchanged; the ApproxNanoflannBackend alias uses 100
// (eps = 0.1) for the high-dimensional regime where exact NN dominates runtime.
// eps = ApproxEpsPermille / 1000.
template <typename T, std::size_t Dim, unsigned ApproxEpsPermille = 0>
class NanoflannBackend {
public:
    using Point = std::array<T, Dim>;

    // nanoflann's eps-approximation factor for this backend (0 => exact).
    static constexpr float search_eps = static_cast<float>(ApproxEpsPermille) / 1000.0f;

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
        params.eps = search_eps;
        (void)index_.radiusSearch(p.data(), radius_sq, matches, params);  // count unused
        out.reserve(out.size() + matches.size());
        for (const auto& m : matches) out.push_back(m.first);
    }

    // Like radius_search, but also appends each neighbor's SQUARED distance to out_sq
    // (parallel to out), reusing the distances nanoflann already computed during the
    // search rather than recomputing them downstream (issue #55). Only meaningful for
    // T == double, where m.second is bit-identical to detail::square_dist; the
    // RadiusSearchWithDists concept restricts the fast path to that case.
    void radius_search_with_dists(const Point& p, T r, std::vector<std::size_t>& out,
                                  std::vector<double>& out_sq) const {
        const T radius_sq = static_cast<T>(r) * static_cast<T>(r);
        thread_local std::vector<nanoflann::ResultItem<std::size_t, T>> matches;
        matches.clear();
        nanoflann::SearchParameters params;
        params.sorted = false;
        params.eps = search_eps;
        (void)index_.radiusSearch(p.data(), radius_sq, matches, params);
        out.reserve(out.size() + matches.size());
        out_sq.reserve(out_sq.size() + matches.size());
        for (const auto& m : matches) { out.push_back(m.first); out_sq.push_back(static_cast<double>(m.second)); }
    }

    // Core-distance via a k-NN query: distance to the min_pts-th nearest point,
    // or nullopt if that neighbor lies beyond r (equivalently, fewer than min_pts
    // points lie within r -- the UNDEFINED core-distance case). O(min_pts log n)
    // instead of scanning the whole eps-neighborhood. For an exact backend this
    // matches the nth_element scan exactly (same kth distance).
    std::optional<double> knn_core_dist(const Point& p, std::size_t min_pts, T r) const {
        thread_local std::vector<std::size_t> idx;
        thread_local std::vector<T> dist_sq;
        idx.resize(min_pts);
        dist_sq.resize(min_pts);
        // findNeighbors (not knnSearch) so the eps-approximation factor applies.
        nanoflann::KNNResultSet<T, std::size_t> result_set(min_pts);
        result_set.init(idx.data(), dist_sq.data());
        nanoflann::SearchParameters params;
        params.eps = search_eps;
        index_.findNeighbors(result_set, p.data(), params);
        const std::size_t found = result_set.size();
        if (found < min_pts) return std::nullopt;
        const T kth_sq = dist_sq[min_pts - 1];
        // Compare in double so a "no cap" sentinel radius (numeric_limits<T>::max(), used by
        // epsilon_estimation_knee) does not overflow r*r. Exact for double T (the pinned Knn path).
        if (static_cast<double>(kth_sq) > static_cast<double>(r) * static_cast<double>(r)) return std::nullopt;
        return std::sqrt(static_cast<double>(kth_sq));
    }

private:
    using Adaptor = detail::PointCloudAdaptor<T, Dim>;
    using Metric = nanoflann::L2_Simple_Adaptor<T, Adaptor>;
    using Index = nanoflann::KDTreeSingleIndexAdaptor<Metric, Adaptor, static_cast<int>(Dim), std::size_t>;

    Adaptor adaptor_;
    Index index_;
};

// Approximate nanoflann backend for the high-dimensional (e.g. 16-D) regime where
// exact nearest-neighbor search dominates the runtime. Same KD-tree, eps-approximate
// queries (default eps = 0.1): faster, with bounded recall loss near the query
// boundary. Swap it in via the Backend template argument of compute_reachability_dists.
template <typename T, std::size_t Dim, unsigned ApproxEpsPermille = 100>
using ApproxNanoflannBackend = NanoflannBackend<T, Dim, ApproxEpsPermille>;

}  // namespace optics

#ifdef OPTICS_ENABLE_BOOST_RTREE
#include "boost_backend.hpp"
#endif
