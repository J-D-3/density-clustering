// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

#include "backend.hpp"
#include "preprocess.hpp"
#include "tree.hpp"
#include "version.hpp"
#include "detail/math.hpp"
#include "detail/thread_pool.hpp"
#include "detail/profile.hpp"
#include "detail/random_projection.hpp"
#include "detail/random_features.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace optics {

typedef std::pair<std::size_t, std::size_t> chi_cluster_indices;
typedef optics::Tree<chi_cluster_indices> cluster_tree;

template <typename T, std::size_t dimension>
using Point = std::array<T, dimension>;


// One entry of the OPTICS cluster-ordering: the point's index plus the
// reachability distance at which it was reached (-1 == UNDEFINED / unreached).
struct reachability_dist {
	reachability_dist( std::size_t point_index_, double reach_dist_ ) : point_index( point_index_ ), reach_dist( reach_dist_ ) {}

	std::string to_string() const {
		return "{" + std::to_string( point_index ) + "," + std::to_string( reach_dist ) + "}";
	}
	std::size_t point_index;
	double reach_dist;
};

inline bool operator<( const reachability_dist& lhs, const reachability_dist& rhs ) {
	return (lhs.reach_dist <= rhs.reach_dist && lhs.reach_dist >= rhs.reach_dist) ? (lhs.point_index < rhs.point_index) : (lhs.reach_dist < rhs.reach_dist);
}
inline bool operator==( const reachability_dist& lhs, const reachability_dist& rhs ) {
	return (lhs.reach_dist <= rhs.reach_dist && lhs.reach_dist >= rhs.reach_dist) && (lhs.point_index == rhs.point_index);
}
inline std::ostringstream& operator<<( std::ostringstream& stream, const reachability_dist& r ) {
	stream << r.to_string();
	return stream;
}


//=== Core OPTICS primitives (internal) ======================================

namespace detail {

// core-distance of a point: nullopt if it has fewer than min_pts neighbors
// (incl. itself), else the distance to its min_pts-th nearest neighbor.
template <typename T, std::size_t Dim>
std::optional<double> compute_core_dist( const Point<T, Dim>& point,
										 const std::vector<Point<T, Dim>>& points,
										 const std::vector<std::size_t>& neighbor_indices,
										 std::size_t min_pts ) {
	if ( neighbor_indices.size() < min_pts ) { return std::nullopt; }

	// Compute each neighbor's squared distance exactly once into a reused buffer,
	// then nth_element on the distances directly. Avoids the per-call index-vector
	// copy and the repeated distance computations a key-comparator would incur.
	// thread_local: the buffer is reused across calls (no allocation after warmup)
	// and stays correct if core_dist is ever invoked from multiple threads.
	thread_local std::vector<double> sq_dists;
	sq_dists.clear();
	sq_dists.reserve( neighbor_indices.size() );
	for ( const std::size_t idx : neighbor_indices ) {
		sq_dists.push_back( detail::square_dist( point, points[idx] ) );
	}
	std::nth_element( sq_dists.begin(), sq_dists.begin() + ( min_pts - 1 ), sq_dists.end() );
	return std::sqrt( sq_dists[min_pts - 1] );
}


// Core-distance from squared distances the backend already computed during the search
// (RadiusSearchWithDists, issue #55), avoiding the recompute in compute_core_dist. The
// nth_element runs on a thread_local copy so the caller's parallel-to-neighbors order is
// preserved (the relaxation indexes into it). For a double backend the kth value is
// bit-identical to compute_core_dist's. nullopt if fewer than min_pts neighbors.
inline std::optional<double> compute_core_dist_from_sq( const std::vector<double>& sq_dists, std::size_t min_pts ) {
	if ( sq_dists.size() < min_pts ) { return std::nullopt; }
	thread_local std::vector<double> scratch;
	scratch.assign( sq_dists.begin(), sq_dists.end() );
	std::nth_element( scratch.begin(), scratch.begin() + ( min_pts - 1 ), scratch.end() );
	return std::sqrt( scratch[min_pts - 1] );
}


// Weighted core-distance (unique-point OPTICS, issue #46). Each neighbor carries an integer
// weight (= how many identical original points it stands for). The neighborhood "count" is the
// SUM of weights within eps (the point itself, at distance 0, is in neighbor_indices and counts
// its full weight); the point is a core object iff that sum reaches min_pts, and the core-distance
// is the distance at which the cumulative weight -- neighbors sorted ascending by distance -- first
// reaches min_pts. This matches scikit-learn DBSCAN's sample_weight semantics. With every weight
// == 1 it reduces exactly to compute_core_dist's min_pts-th-neighbor distance. Unlike the
// unweighted path it cannot use nth_element: the k-th position depends on a running prefix sum, so
// the distances are fully sorted. nullopt if the total neighborhood weight is below min_pts.
template <typename T, std::size_t Dim>
std::optional<double> compute_core_dist_weighted( const Point<T, Dim>& point,
												  const std::vector<Point<T, Dim>>& points,
												  const std::vector<std::size_t>& neighbor_indices,
												  const std::vector<std::size_t>& weights,
												  std::size_t min_pts ) {
	std::size_t total = 0;
	for ( const std::size_t idx : neighbor_indices ) { total += weights[idx]; }
	if ( total < min_pts ) { return std::nullopt; }

	thread_local std::vector<std::pair<double, std::size_t>> dw;  // (squared distance, weight)
	dw.clear();
	dw.reserve( neighbor_indices.size() );
	for ( const std::size_t idx : neighbor_indices ) {
		dw.emplace_back( detail::square_dist( point, points[idx] ), weights[idx] );
	}
	std::sort( dw.begin(), dw.end(), []( const auto& a, const auto& b ) { return a.first < b.first; } );
	std::size_t acc = 0;
	for ( const auto& [sq, w] : dw ) {
		acc += w;
		if ( acc >= min_pts ) { return std::sqrt( sq ); }
	}
	return std::sqrt( dw.back().first );  // unreachable: total >= min_pts guarantees an earlier return
}


// Weighted core-distance from squared distances the backend already computed (the #55 reuse path
// and sOPTICS). sq_dists is parallel to neighbor_indices; weights is indexed by point index. Same
// weighted-selection semantics as compute_core_dist_weighted.
inline std::optional<double> compute_core_dist_weighted_from_sq( const std::vector<double>& sq_dists,
																 const std::vector<std::size_t>& neighbor_indices,
																 const std::vector<std::size_t>& weights,
																 std::size_t min_pts ) {
	std::size_t total = 0;
	for ( const std::size_t idx : neighbor_indices ) { total += weights[idx]; }
	if ( total < min_pts ) { return std::nullopt; }

	thread_local std::vector<std::pair<double, std::size_t>> dw;
	dw.clear();
	dw.reserve( sq_dists.size() );
	for ( std::size_t j = 0; j < sq_dists.size(); ++j ) {
		dw.emplace_back( sq_dists[j], weights[neighbor_indices[j]] );
	}
	std::sort( dw.begin(), dw.end(), []( const auto& a, const auto& b ) { return a.first < b.first; } );
	std::size_t acc = 0;
	for ( const auto& [sq, w] : dw ) {
		acc += w;
		if ( acc >= min_pts ) { return std::sqrt( sq ); }
	}
	return std::sqrt( dw.back().first );
}


// Min-heap order for the seed priority queue: smallest reachability first, ties
// broken by smallest point index (matching reachability_dist's operator<). The
// queue uses lazy deletion -- "decrease-key" pushes a new (smaller) entry and
// the authoritative value lives in the reachability[] array; stale pops are
// skipped. This reproduces the std::set pop order exactly while avoiding a node
// allocation per insert.
struct seed_min_heap_cmp {
	bool operator()( const reachability_dist& a, const reachability_dist& b ) const { return b < a; }
};

using seed_queue = std::priority_queue<reachability_dist, std::vector<reachability_dist>, seed_min_heap_cmp>;


// Drives the OPTICS cluster-ordering from two providers, so the loop is shared by
// OPTICS and sOPTICS. The neighbor/core-distance policy (backend + epsilon, Scan/Knn,
// random projections, ...) lives entirely in the providers; this function owns only the
// algorithm-agnostic machinery: the seed priority queue, the relaxation, lazy deletion,
// and the ordered-list assembly.
//   neighbors_of(idx)       -> const std::vector<std::size_t>&  : idx's neighbor indices.
//        CONTRACT: the returned reference stays valid only until the next neighbors_of
//        call, so an OnDemand provider may hand back a single reused buffer.
//   core_dist_of(idx, nbrs) -> std::optional<double>            : idx's core distance,
//        or nullopt when UNDEFINED (fewer than min_pts neighbors).
//   dist_of(idx, j, o)      -> double  : distance between point idx and its j-th neighbor
//        o = nbrs[j]. The default callers recompute detail::dist(points[idx], points[o]);
//        a RadiusSearchWithDists backend instead returns the search's reused distance (#55).
//        For the recompute default this is byte-identical to the previous inline call.
//   prof : phase profiler; the relax and loop phases are accumulated here. The caller
//        times index_build/precompute and calls prof.report() after this returns.
template <class T, std::size_t Dim, class NeighborsOf, class CoreDistOf, class DistOf>
std::vector<reachability_dist> optics_order(
		const std::vector<Point<T, Dim>>& points,
		NeighborsOf&& neighbors_of, CoreDistOf&& core_dist_of, DistOf&& dist_of, PhaseProfiler& prof ) {
	const std::size_t n = points.size();

	std::vector<char> processed( n, 0 );
	std::vector<std::size_t> ordered_list;
	ordered_list.reserve( n );
	std::vector<double> reachability( n, -1.0 );

	// Reused across points; drained to empty whenever an expansion finishes.
	seed_queue seeds;
	const auto relax_neighbors = [&]( std::size_t idx, const std::vector<std::size_t>& nbrs, double core_dist ) {
		[[maybe_unused]] auto _s = prof.scope( prof.relax );
		for ( std::size_t j = 0; j < nbrs.size(); ++j ) {
			const std::size_t o = nbrs[j];
			if ( processed[o] ) { continue; }
			const double new_rd = std::max( core_dist, dist_of( idx, j, o ) );
			if ( reachability[o] < 0.0 || new_rd < reachability[o] ) {
				reachability[o] = new_rd;  // authoritative; the prior heap entry becomes stale
				seeds.push( reachability_dist( o, new_rd ) );
			}
		}
	};

	const auto _t_loop = prof.now();
	for ( std::size_t point_idx = 0; point_idx < n; point_idx++ ) {
		if ( processed[point_idx] ) { continue; }
		processed[point_idx] = 1;
		ordered_list.push_back( point_idx );

		const auto& neighbor_indices = neighbors_of( point_idx );
		const auto core_dist = core_dist_of( point_idx, neighbor_indices );
		if ( core_dist.has_value() ) { relax_neighbors( point_idx, neighbor_indices, *core_dist ); }

		while ( !seeds.empty() ) {
			const reachability_dist s = seeds.top();
			seeds.pop();
			// Lazy deletion: skip entries already processed or superseded by a smaller push.
			if ( processed[s.point_index] || s.reach_dist != reachability[s.point_index] ) { continue; }
			processed[s.point_index] = 1;
			ordered_list.push_back( s.point_index );

			const auto& s_neighbor_indices = neighbors_of( s.point_index );
			const auto s_core_dist = core_dist_of( s.point_index, s_neighbor_indices );
			if ( s_core_dist.has_value() ) { relax_neighbors( s.point_index, s_neighbor_indices, *s_core_dist ); }
		}
	}
	assert( ordered_list.size() == n );
	prof.add( prof.loop, _t_loop );

	std::vector<reachability_dist> result;
	result.reserve( n );
	for ( const std::size_t point_idx : ordered_list ) {
		result.emplace_back( point_idx, reachability[point_idx] );
	}
	return result;
}

}  // namespace detail


//=== Epsilon estimation =====================================================

namespace detail {

template <typename T, std::size_t dimension>
std::pair<Point<T, dimension>, Point<T, dimension>> bounding_box( const std::vector<Point<T, dimension>>& points ) {
	assert( points.size() > 0 );  // Bounding box of 0 points not defined
	static_assert( std::is_convertible<T, double>::value, "bounding_box(): point type must be convertible to double" );

	Point<T, dimension> min( points[0] );
	Point<T, dimension> max( points[0] );
	for ( const auto& p : points ) {
		for ( std::size_t i = 0; i < dimension; i++ ) {
			if ( p[i] < min[i] ) min[i] = p[i];
			if ( p[i] > max[i] ) max[i] = p[i];
		}
	}
	return { min, max };
}

template <typename T, std::size_t dimension>
double hypercuboid_volume( const Point<T, dimension>& bl, const Point<T, dimension>& tr ) {
	double volume = 1.0;
	for ( std::size_t i = 0; i < dimension; i++ ) {
		volume *= std::abs( static_cast<double>( tr[i] - bl[i] ) );
	}
	return volume;
}

}  // namespace detail

// Heuristic generating distance: the expected MinPts-nearest-neighbor distance
// assuming a uniform point distribution over the data's bounding box.
template <typename T, std::size_t dimension>
double epsilon_estimation( const std::vector<Point<T, dimension>>& points, std::size_t min_pts ) {
	static_assert( std::is_convertible<double, T>::value, "epsilon_estimation: point type 'T' must be convertible to double" );
	static_assert( dimension >= 1, "epsilon_estimation: dimension must be >= 1" );
	if ( points.size() <= 1 ) { return 0.0; }

	const auto space = detail::bounding_box( points );
	// Use only dimensions with non-zero extent, so degenerate inputs (collinear,
	// planar, or all-identical) yield a sensible scale instead of a zero volume.
	double effective_volume = 1.0;
	std::size_t d_eff = 0;
	for ( std::size_t i = 0; i < dimension; ++i ) {
		const double extent = std::abs( static_cast<double>( space.second[i] - space.first[i] ) );
		if ( extent > 0.0 ) {
			effective_volume *= extent;
			++d_eff;
		}
	}
	// d_eff == 0 means every coordinate is identical across all points: there is no
	// spatial scale, every pairwise distance is 0, and any positive radius groups
	// them identically, so the concrete value is immaterial.
	if ( d_eff == 0 ) { return 1.0; }

	const double d = static_cast<double>( d_eff );
	const double space_per_minpts_points = ( effective_volume / static_cast<double>( points.size() ) ) * static_cast<double>( min_pts );
	const double n_dim_unit_ball_vol = std::sqrt( std::pow( detail::pi, d ) ) / std::tgamma( d / 2.0 + 1.0 );
	return std::pow( space_per_minpts_points / n_dim_unit_ball_vol, 1.0 / d );
}

// Weighted variant for unique-point OPTICS (issue #46): the deduplicated cloud has the same
// bounding-box geometry as the full cloud, only fewer (weighted) points, so the density must use
// the TOTAL weight (= the original point count) rather than the unique-point count. With every
// weight == 1 this equals epsilon_estimation(points, min_pts). weights is parallel to points.
template <typename T, std::size_t dimension>
double epsilon_estimation( const std::vector<Point<T, dimension>>& points, std::size_t min_pts,
						   const std::vector<std::size_t>& weights ) {
	static_assert( std::is_convertible<double, T>::value, "epsilon_estimation: point type 'T' must be convertible to double" );
	static_assert( dimension >= 1, "epsilon_estimation: dimension must be >= 1" );
	if ( points.size() <= 1 ) { return 0.0; }

	double total_weight = 0.0;
	for ( const std::size_t w : weights ) { total_weight += static_cast<double>( w ); }
	if ( total_weight <= 0.0 ) { return epsilon_estimation( points, min_pts ); }

	const auto space = detail::bounding_box( points );
	double effective_volume = 1.0;
	std::size_t d_eff = 0;
	for ( std::size_t i = 0; i < dimension; ++i ) {
		const double extent = std::abs( static_cast<double>( space.second[i] - space.first[i] ) );
		if ( extent > 0.0 ) {
			effective_volume *= extent;
			++d_eff;
		}
	}
	if ( d_eff == 0 ) { return 1.0; }

	const double d = static_cast<double>( d_eff );
	const double space_per_minpts_points = ( effective_volume / total_weight ) * static_cast<double>( min_pts );
	const double n_dim_unit_ball_vol = std::sqrt( std::pow( detail::pi, d ) ) / std::tgamma( d / 2.0 + 1.0 );
	return std::pow( space_per_minpts_points / n_dim_unit_ball_vol, 1.0 / d );
}


namespace detail {
// Kneedle: given a set of k-distances, return the "knee" -- the value at the index of maximum
// perpendicular distance to the chord joining the first and last (sorted) sample. Below the knee
// points sit inside dense regions, above it they are sparse/noise, so the knee is the within-cluster
// scale. Sorts k_dist in place. Returns <= 0.0 to signal a degenerate curve (empty, or all-zero),
// so the caller can fall back to the uniform-density estimate (auto-eps must never collapse to 0).
inline double kneedle( std::vector<double>& k_dist ) {
	if ( k_dist.size() < 3 ) { return k_dist.empty() ? -1.0 : k_dist.back(); }
	std::sort( k_dist.begin(), k_dist.end() );  // ascending
	const std::size_t m = k_dist.size();
	const double y0 = k_dist.front();
	const double dx = static_cast<double>( m - 1 );
	const double dy = k_dist.back() - y0;
	const double norm = std::sqrt( dx * dx + dy * dy );
	if ( norm <= 0.0 ) { return k_dist.back(); }  // flat curve: that value (0 => caller falls back)
	double best_metric = -1.0;
	std::size_t best_i = m - 1;
	for ( std::size_t i = 0; i < m; ++i ) {
		const double d = std::abs( dy * static_cast<double>( i ) - dx * ( k_dist[i] - y0 ) ) / norm;
		if ( d > best_metric ) { best_metric = d; best_i = i; }
	}
	return k_dist[best_i];
}
}  // namespace detail

// Alternative generating-distance heuristic: the k-distance knee (the classic DBSCAN
// rule of thumb). Unlike epsilon_estimation, which assumes a uniform density over the
// bounding box and so over-estimates epsilon on clustered data, this looks at the actual
// k-nearest-neighbor (k = min_pts) distances and takes the "knee" (see detail::kneedle).
//
// Opt-in: callers pass the result as `epsilon` to compute_reachability_dists; the default
// path is unchanged. Requires a backend modeling KnnCoreDist (e.g. NanoflannBackend).
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
double epsilon_estimation_knee( const std::vector<std::array<T, Dim>>& points, std::size_t min_pts ) {
	static_assert( std::is_floating_point_v<T>, "epsilon_estimation_knee: coordinate type 'T' must be float or double" );
	static_assert( Dim >= 1, "epsilon_estimation_knee: dimension must be >= 1" );
	static_assert( KnnCoreDist<Backend, T, Dim>,
		"epsilon_estimation_knee requires a backend modeling KnnCoreDist (e.g. NanoflannBackend)" );
	if ( min_pts < 1 ) { throw std::invalid_argument( "epsilon_estimation_knee: min_pts must be >= 1" ); }
	// Too few points for a k-distance curve: defer to the uniform-density heuristic.
	if ( points.size() <= min_pts ) { return epsilon_estimation( points, min_pts ); }

	// Collect each point's k-distance (distance to its min_pts-th NN). A huge radius makes
	// the backend's radius cap inert, so this is a plain k-NN distance for every point.
	const Backend backend( points );
	const T no_cap = std::numeric_limits<T>::max();
	std::vector<double> k_dist;
	k_dist.reserve( points.size() );
	for ( const auto& p : points ) {
		const auto cd = backend.knn_core_dist( p, min_pts, no_cap );
		if ( cd.has_value() ) { k_dist.push_back( *cd ); }
	}
	const double knee = detail::kneedle( k_dist );
	return knee > 0.0 ? knee : epsilon_estimation( points, min_pts );
}

// Weighted-knee epsilon for unique-point OPTICS (issue #46): the knee of the WEIGHTED k-distances
// (the distance at which each point's cumulative neighbor weight reaches min_pts), so the estimate
// tracks the within-cluster scale on a deduplicated cloud instead of the uniform-density over-shoot.
// Each weight is >= 1, so the weighted k-distance is found within the min_pts nearest points -- the
// per-point cost matches the unweighted knee. Requires a backend modeling KnnCoreDistWeighted.
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
double epsilon_estimation_knee( const std::vector<std::array<T, Dim>>& points, std::size_t min_pts,
								const std::vector<std::size_t>& weights ) {
	static_assert( std::is_floating_point_v<T>, "epsilon_estimation_knee: coordinate type 'T' must be float or double" );
	static_assert( Dim >= 1, "epsilon_estimation_knee: dimension must be >= 1" );
	static_assert( KnnCoreDistWeighted<Backend, T, Dim>,
		"weighted epsilon_estimation_knee requires a backend modeling KnnCoreDistWeighted (e.g. NanoflannBackend)" );
	if ( min_pts < 1 ) { throw std::invalid_argument( "epsilon_estimation_knee: min_pts must be >= 1" ); }
	if ( weights.size() != points.size() ) { throw std::invalid_argument( "epsilon_estimation_knee: weights.size() must equal points.size()" ); }
	double total_weight = 0.0;
	for ( const std::size_t w : weights ) { total_weight += static_cast<double>( w ); }
	// Too little total weight for a k-distance curve: defer to the (weighted) uniform heuristic.
	if ( points.size() <= 1 || total_weight <= static_cast<double>( min_pts ) ) { return epsilon_estimation( points, min_pts, weights ); }

	const Backend backend( points );
	std::vector<double> k_dist;
	k_dist.reserve( points.size() );
	for ( const auto& p : points ) {
		const auto cd = backend.knn_core_dist_weighted( p, weights, min_pts );
		if ( cd.has_value() ) { k_dist.push_back( *cd ); }
	}
	const double knee = detail::kneedle( k_dist );
	return knee > 0.0 ? knee : epsilon_estimation( points, min_pts, weights );
}


//=== The OPTICS ordering ====================================================

// Computes the OPTICS cluster-ordering and reachability distances.
//   T, Dim   : coordinate type (float/double) and dimensionality (deduced from points).
//   Backend  : neighbor-search backend; defaults to nanoflann.
//   epsilon  : generating distance; auto-estimated when <= 0.
//   mode     : OnDemand (default) queries one neighborhood at a time during the
//              ordering -- O(one neighborhood) memory, and on dense clouds it is also
//              faster (it avoids materializing/re-reading a huge neighbor cache).
//              Precompute caches every neighborhood up front, in parallel: faster on
//              sparse/low-density clouds, but uses O(n * avg_neighbors) memory (which
//              can be tens of GB on dense data). Both yield identical results.
//   n_threads: worker threads for the Precompute query phase (0 => hardware
//              concurrency). Ignored by OnDemand (the ordering loop is sequential).
//   core_dist: Scan (default) or Knn core-distance computation. Knn is faster on
//              dense clouds and yields identical results; it falls back to Scan
//              for backends that do not model KnnCoreDist.
//   max_precompute_bytes: optional guard for the Precompute neighbor cache. When > 0,
//              the cache size is estimated from a small sample before allocating; if the
//              estimate exceeds this many bytes, throws std::runtime_error suggesting
//              OnDemand instead of attempting a multi-GB allocation. 0 (default) = no check;
//              ignored by OnDemand (which has no such buffer).
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
std::vector<reachability_dist> compute_reachability_dists(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts,
	double epsilon = -1.0, NeighborMode mode = NeighborMode::OnDemand, unsigned n_threads = 0,
	CoreDistMode core_dist_mode = CoreDistMode::Scan, std::size_t max_precompute_bytes = 0,
	const std::vector<std::size_t>& weights = {} ) {

	static_assert( std::is_floating_point_v<T>, "compute_reachability_dists: coordinate type 'T' must be float or double" );
	static_assert( Dim >= 1, "compute_reachability_dists: dimension must be >= 1" );
	static_assert( NeighborSearch<Backend, T, Dim>, "Backend does not satisfy the NeighborSearch concept" );

	if ( min_pts < 1 ) { throw std::invalid_argument( "compute_reachability_dists: min_pts must be >= 1" ); }
	if ( points.empty() ) { return {}; }

	// Unique-point / weighted OPTICS (issue #46). When weights is non-empty each point stands for
	// `weights[i]` identical original points, and the core-distance / neighborhood-count become
	// weight-aware (see compute_core_dist_weighted). Empty weights => the unweighted path, byte for
	// byte unchanged.
	const bool weighted = !weights.empty();
	if ( weighted && weights.size() != points.size() ) {
		throw std::invalid_argument( "compute_reachability_dists: weights.size() must equal points.size()" );
	}

	double eps = epsilon;
	if ( eps <= 0.0 ) {
		// Auto-epsilon. Default to the k-distance-knee estimator, which tracks the actual
		// within-cluster scale -- the uniform-density epsilon_estimation over-estimates on
		// clustered data, which both slows the dense path AND smooths the reachability so Xi
		// under-segments (issue #57). The knee needs a KnnCoreDist backend; for backends that
		// lack it (e.g. Boost) fall back to the uniform estimate (if constexpr => zero overhead,
		// and no knn_core_dist instantiation for backends without it). Both yield a positive
		// scale for any >= 2-point input (degenerate inputs included), so no zero-radius collapse.
		// Weighted mode: prefer the WEIGHTED k-distance knee (tracks the within-cluster scale on a
		// deduplicated cloud); fall back to the total-weight uniform estimate for backends that can't
		// answer a weighted k-distance.
		if ( weighted ) {
			if constexpr ( KnnCoreDistWeighted<Backend, T, Dim> ) {
				eps = epsilon_estimation_knee<T, Dim, Backend>( points, min_pts, weights );
			} else {
				eps = epsilon_estimation( points, min_pts, weights );
			}
		} else if constexpr ( KnnCoreDist<Backend, T, Dim> ) {
			eps = epsilon_estimation_knee<T, Dim, Backend>( points, min_pts );
		} else {
			eps = epsilon_estimation( points, min_pts );
		}
	}
	const T eps_t = static_cast<T>( eps );

	// Optional phase profiler: a no-op unless built with -DOPTICS_PROFILE (see
	// detail/profile.hpp). The hooks below carry no #ifdef of their own.
	detail::PhaseProfiler _prof;
	const auto _t_build = _prof.now();
	const Backend backend( points );
	_prof.add( _prof.index_build, _t_build );
	const std::size_t n = points.size();

	// #55: a backend modeling RadiusSearchWithDists (double coordinates only, where its
	// squared distances are bit-identical to detail::square_dist) lets us reuse the search's
	// distances for the core-distance scan and the relaxation instead of recomputing them --
	// a win on dense clouds, where neighborhood processing dominates. cur_sq points at the
	// squared distances of the point currently being expanded (parallel to its neighbor
	// list), valid until the next neighbors_of call.
	constexpr bool reuse_dists = RadiusSearchWithDists<Backend, T, Dim>;
	std::vector<std::vector<double>> neighbor_sq;   // parallel to `neighbors` (Precompute path)
	const std::vector<double>* cur_sq = nullptr;

	// Neighbor acquisition: precompute-all (parallel) or on-demand.
	const auto _t_pre = _prof.now();
	std::vector<std::vector<std::size_t>> neighbors;
	if ( mode == NeighborMode::Precompute ) {
		// Optional guard: estimate the neighbor-cache size from a small sample before
		// committing to it, so a dense cloud doesn't silently try to allocate tens of GB.
		// The sampling cost is incurred only when a cap is set (default 0 == no check).
		if ( max_precompute_bytes > 0 ) {
			const std::size_t sample = std::min<std::size_t>( n, 64 );
			std::size_t neighbor_sum = 0;
			std::vector<std::size_t> probe;
			for ( std::size_t s = 0; s < sample; ++s ) {
				probe.clear();
				backend.radius_search( points[( n / sample ) * s], eps_t, probe );
				neighbor_sum += probe.size();
			}
			const double avg_neighbors = static_cast<double>( neighbor_sum ) / static_cast<double>( sample );
			const double est_bytes = static_cast<double>( n ) * static_cast<double>( sizeof( std::vector<std::size_t> ) )
								   + static_cast<double>( n ) * avg_neighbors * static_cast<double>( sizeof( std::size_t ) );
			if ( est_bytes > static_cast<double>( max_precompute_bytes ) ) {
				std::ostringstream msg;
				msg << "compute_reachability_dists: estimated Precompute neighbor cache ~"
					<< static_cast<std::size_t>( est_bytes / ( 1024.0 * 1024.0 ) ) << " MB ("
					<< n << " points, avg " << avg_neighbors << " neighbors/point) exceeds the "
					<< ( max_precompute_bytes / ( 1024 * 1024 ) ) << " MB cap. Use "
					   "NeighborMode::OnDemand (O(one neighborhood) memory) or a smaller epsilon.";
				throw std::runtime_error( msg.str() );
			}
		}
		neighbors.resize( n );
		if constexpr ( reuse_dists ) { neighbor_sq.resize( n ); }
		detail::parallel_for( n_threads, n, [&]( std::size_t i ) {
			if constexpr ( reuse_dists ) { backend.radius_search_with_dists( points[i], eps_t, neighbors[i], neighbor_sq[i] ); }
			else { backend.radius_search( points[i], eps_t, neighbors[i] ); }
		} );
	}
	_prof.add( _prof.precompute, _t_pre );
	std::vector<std::size_t> ondemand_buf;
	std::vector<double> ondemand_sq;
	const auto neighbors_of = [&]( std::size_t idx ) -> const std::vector<std::size_t>& {
		if ( mode == NeighborMode::Precompute ) {
			if constexpr ( reuse_dists ) { cur_sq = &neighbor_sq[idx]; }
			return neighbors[idx];
		}
		ondemand_buf.clear();
		if constexpr ( reuse_dists ) {
			ondemand_sq.clear();
			backend.radius_search_with_dists( points[idx], eps_t, ondemand_buf, ondemand_sq );
			cur_sq = &ondemand_sq;
		} else {
			backend.radius_search( points[idx], eps_t, ondemand_buf );
		}
		return ondemand_buf;
	};

	// Core-distance of a point: the Knn path queries the backend directly (cheap on
	// dense clouds); both paths return identical values. if constexpr keeps the Knn
	// branch out of instantiations for backends without the capability, which then
	// always Scan.
	const auto core_dist_of = [&]( std::size_t idx, [[maybe_unused]] const std::vector<std::size_t>& nbrs ) -> std::optional<double> {
		[[maybe_unused]] auto _s = _prof.scope( _prof.core_dist );
		(void)core_dist_mode;  // unused when the backend lacks a knn_core_dist capability
		// Weighted mode: the knn_core_dist fast path fetches exactly min_pts neighbors, which is
		// wrong once neighbors carry weights (the min_pts-th cumulative-weight neighbor may be the
		// 1st or the 100th), so weighted runs always Scan the eps-neighborhood. CoreDistMode::Knn
		// silently downgrades to Scan here.
		if ( weighted ) {
			if constexpr ( reuse_dists ) { return detail::compute_core_dist_weighted_from_sq( *cur_sq, nbrs, weights, min_pts ); }
			else { return detail::compute_core_dist_weighted( points[idx], points, nbrs, weights, min_pts ); }
		}
		if constexpr ( KnnCoreDist<Backend, T, Dim> ) {
			if ( core_dist_mode == CoreDistMode::Knn ) {
				return backend.knn_core_dist( points[idx], min_pts, eps_t );
			}
		}
		if constexpr ( reuse_dists ) { (void)idx; return detail::compute_core_dist_from_sq( *cur_sq, min_pts ); }
		else { return detail::compute_core_dist( points[idx], points, nbrs, min_pts ); }
	};

	// Pairwise distance for the relaxation: reuse the search's squared distance (#55) when
	// available, else recompute. For the recompute default this is byte-identical to the
	// previous inline detail::dist call, so OPTICS orderings are unchanged.
	const auto dist_of = [&]( [[maybe_unused]] std::size_t idx, [[maybe_unused]] std::size_t j,
							  [[maybe_unused]] std::size_t o ) -> double {
		if constexpr ( reuse_dists ) { return std::sqrt( ( *cur_sq )[j] ); }
		else { return detail::dist( points[idx], points[o] ); }
	};

	// All neighbor/core-distance policy is now captured in the closures above; the
	// algorithm-agnostic ordering machinery lives in detail::optics_order (shared with
	// sOPTICS). index_build/precompute were timed above; the helper accumulates the
	// relax/loop phases into _prof, and we report once it returns.
	auto result = detail::optics_order<T, Dim>( points, neighbors_of, core_dist_of, dist_of, _prof );
	_prof.report( n );
	return result;
}


//=== sOPTICS: scalable approximate OPTICS via random projections ============

// sOPTICS (Xu & Pham, NeurIPS 2024): a scalable, approximate OPTICS that replaces the
// exact radius search with CEOs random-projection neighborhoods (see
// detail/random_projection.hpp). It returns the same reachability_dist cluster-ordering
// as compute_reachability_dists, so all existing extraction (get_cluster_indices,
// get_chi_clusters, cluster_threshold, ...) applies unchanged.
//
// Metric is COSINE: points are L2-normalized onto the unit sphere internally, where
// Euclidean distance is monotone in cosine distance, so the ordering / reachability-plot
// valleys match the cosine ones. Do NOT expect raw-Euclidean OPTICS semantics on
// un-normalized data (other metrics are issue #51). The result is randomized but
// deterministic in `seed` (identical seed => identical ordering); validate via
// statistical agreement (Rand/NMI) with exact OPTICS, not bit-identical orderings.
//
//   epsilon       : generating distance -- a Euclidean radius on the unit sphere, in
//                   [0, 2]. Like OPTICS it bounds the (approximate) neighborhoods; set it
//                   generously to reveal structure across scales. <= 0 => 2.0 (keep all
//                   CEOs candidates; the candidate pool is bounded regardless).
//   n_projections : D Gaussian random vectors (more => higher recall, more time).
//   k, m          : CEOs tunables (top-k extreme vectors/point; top-m extreme points/
//                   vector). 0 => defaults (k = 10, m = 2*min_pts).
//   seed          : RNG seed for the projections (determinism).
//   n_threads     : workers for the parallel projection/candidate phases (0 => hw).
//   metric        : which geometry the clustering should reflect (issue #51). Cosine
//                   (default) is the native CEOs metric. L2 / L1 embed the points into
//                   random Fourier features whose cosine similarity approximates the
//                   Gaussian / Laplacian kernel (detail/random_features.hpp), then run the
//                   cosine pipeline on the features -- so the ordering tracks Euclidean /
//                   Manhattan distance on the ORIGINAL data. (chi^2 / JS: not yet, see #51.)
//   kernel_scale  : kernel bandwidth (sigma) for L2 / L1; <= 0 => an auto median-distance
//                   heuristic. Ignored for Cosine.
//   projection    : CEOs projection backend (issue #58). Gaussian (default) dots each point with
//                   D explicit N(0,1) vectors. Structured uses FHT "spinners" -- O(D log Dim) per
//                   point instead of O(D*Dim), at the cost of materializing the n x D table -- which
//                   lowers the projection overhead that makes sOPTICS slower than exact OPTICS at
//                   small scale. Approximate either way (validated by Rand agreement, not bit-identity).
template <class T, std::size_t Dim>
std::vector<reachability_dist> compute_soptics_reachability_dists(
		const std::vector<std::array<T, Dim>>& points, std::size_t min_pts,
		double epsilon = -1.0, unsigned n_projections = 1024, unsigned k = 0,
		std::size_t m = 0, unsigned seed = 42, unsigned n_threads = 0,
		Metric metric = Metric::Cosine, double kernel_scale = 0.0,
		const std::vector<std::size_t>& weights = {},
		SopticsProjection projection = SopticsProjection::Gaussian ) {

	static_assert( std::is_floating_point_v<T>, "compute_soptics_reachability_dists: coordinate type 'T' must be float or double" );
	static_assert( Dim >= 1, "compute_soptics_reachability_dists: dimension must be >= 1" );
	if ( min_pts < 1 ) { throw std::invalid_argument( "compute_soptics_reachability_dists: min_pts must be >= 1" ); }
	if ( points.empty() ) { return {}; }

	// Unique-point / weighted sOPTICS (issue #46). Same semantics as weighted OPTICS: the
	// approximate core-distance becomes the distance at which the cumulative neighbor weight reaches
	// min_pts. CEOs candidate generation is geometrically weight-independent (it operates on the
	// unique points), and its default m = 2*min_pts UNIQUE candidates per vector only ever
	// over-satisfies min_pts in weight terms, so no change is needed there. Empty weights => the
	// unweighted path, unchanged. NOTE: exact dedup (upstream) merges bit-identical raw coords; two
	// points that are scalar multiples (cosine-identical after normalization) are NOT merged.
	const bool weighted = !weights.empty();
	if ( weighted && weights.size() != points.size() ) {
		throw std::invalid_argument( "compute_soptics_reachability_dists: weights.size() must equal points.size()" );
	}

	// Non-cosine metric: embed into random Fourier features whose cosine geometry
	// approximates the target kernel, then run the cosine pipeline on the features. The
	// result's point indices line up 1:1 with the input, so extraction (and weights) are unchanged.
	if ( metric != Metric::Cosine ) {
		constexpr std::size_t FeatDim = 256;  // 128 random frequencies (cos/sin pairs)
		const double sigma = ( kernel_scale > 0.0 ) ? kernel_scale : detail::auto_kernel_scale( points, metric );
		const auto feats = detail::embed_random_features<FeatDim, T, Dim>( points, metric, sigma, seed, n_threads );
		return compute_soptics_reachability_dists<double, FeatDim>(
			feats, min_pts, epsilon, n_projections, k, m, seed, n_threads, Metric::Cosine, 0.0, weights, projection );
	}

	// L2-normalize onto the unit sphere (cosine metric). A zero-norm point (the origin)
	// has no direction; leave it unchanged -- it is a degenerate input either way.
	std::vector<std::array<T, Dim>> unit( points.size() );
	for ( std::size_t i = 0; i < points.size(); ++i ) {
		double nrm_sq = 0.0;
		for ( std::size_t c = 0; c < Dim; ++c ) {
			const double v = static_cast<double>( points[i][c] );
			nrm_sq += v * v;
		}
		const double nrm = std::sqrt( nrm_sq );
		if ( nrm > 0.0 ) {
			for ( std::size_t c = 0; c < Dim; ++c ) { unit[i][c] = static_cast<T>( static_cast<double>( points[i][c] ) / nrm ); }
		} else {
			unit[i] = points[i];
		}
	}

	// On the unit sphere all Euclidean distances lie in [0, 2]; the permissive default
	// keeps every CEOs candidate (the pool is bounded by ~2*k*m anyway).
	const double eps = ( epsilon <= 0.0 ) ? 2.0 : epsilon;

	detail::CeosParams params;
	params.n_projections = n_projections;
	params.k = k;
	params.m = m;
	params.seed = seed;
	params.n_threads = n_threads;
	params.projection = ( projection == SopticsProjection::Structured )
		? detail::CeosParams::Projection::Structured
		: detail::CeosParams::Projection::Gaussian;
	// The CEOs index returns each candidate's squared distance (computed in its eps-filter)
	// alongside the neighbor list, so the core-distance scan and the relaxation reuse them
	// instead of recomputing detail::square_dist -- the sOPTICS analogue of #55. All these
	// distances come from detail::square_dist, so the ordering is byte-identical to the
	// recompute path (no double-only gate needed; cur_sq points at the current point's
	// squared distances, parallel to its neighbor list).
	std::vector<std::vector<double>> neighbor_sq;
	const auto neighbors = detail::ceos_neighbors( unit, eps, min_pts, params, &neighbor_sq );

	detail::PhaseProfiler prof;
	const std::vector<double>* cur_sq = nullptr;
	const auto neighbors_of = [&]( std::size_t idx ) -> const std::vector<std::size_t>& {
		cur_sq = &neighbor_sq[idx];
		return neighbors[idx];
	};
	const auto core_dist_of = [&]( std::size_t /*idx*/, const std::vector<std::size_t>& nbrs ) -> std::optional<double> {
		if ( weighted ) { return detail::compute_core_dist_weighted_from_sq( *cur_sq, nbrs, weights, min_pts ); }
		return detail::compute_core_dist_from_sq( *cur_sq, min_pts );
	};
	const auto dist_of = [&]( std::size_t /*idx*/, std::size_t j, std::size_t /*o*/ ) -> double {
		return std::sqrt( ( *cur_sq )[j] );
	};
	auto result = detail::optics_order<T, Dim>( unit, neighbors_of, core_dist_of, dist_of, prof );
	prof.report( points.size() );
	return result;
}


//=== Threshold (DBSCAN-style) cluster extraction ============================

// Cuts the reachability plot at a threshold: contiguous runs below it become
// clusters; points at/above it (and unreached points) become singletons.
inline std::vector<std::vector<std::size_t>> get_cluster_indices( const std::vector<reachability_dist>& reach_dists, double reachability_threshold ) {
	assert( reach_dists.empty() || reach_dists.front().reach_dist < 0.0 );
	std::vector<std::vector<std::size_t>> result;
	for ( const auto& r : reach_dists ) {
		if ( r.reach_dist < 0.0 || r.reach_dist >= reachability_threshold ) {
			result.push_back( { r.point_index } );
		} else {
			result.back().push_back( r.point_index );
		}
	}
	return result;
}

template <typename T, std::size_t dimension>
std::vector<std::vector<std::array<T, dimension>>> get_cluster_points(
	const std::vector<reachability_dist>& reach_dists, double reachability_threshold,
	const std::vector<std::array<T, dimension>>& points ) {
	const auto clusters = get_cluster_indices( reach_dists, reachability_threshold );
	std::vector<std::vector<std::array<T, dimension>>> result;
	result.reserve( clusters.size() );
	for ( const auto& cluster_indices : clusters ) {
		std::vector<std::array<T, dimension>> group;
		group.reserve( cluster_indices.size() );
		for ( const std::size_t idx : cluster_indices ) { group.push_back( points[idx] ); }
		result.push_back( std::move( group ) );
	}
	return result;
}


//=== Xi / steep-area cluster extraction =====================================

namespace detail {
// A steep-down area (start/end index + maximum-in-between value); internal to
// the Xi extraction.
struct SDA {
	SDA( std::size_t begin_idx_, std::size_t end_idx_, double mib_ ) : begin_idx( begin_idx_ ), end_idx( end_idx_ ), mib( mib_ ) {}
	std::size_t begin_idx;
	std::size_t end_idx;
	double mib;
};
}  // namespace detail


inline std::vector<chi_cluster_indices> get_chi_clusters_flat( const std::vector<reachability_dist>& reach_dists_, const double chi, std::size_t min_pts, double steep_area_min_diff = 0.0, std::size_t min_cluster_size = 0, const std::vector<std::size_t>& position_weights = {} ) {
	// The Xi extractor's steep-area span cap and minimum cluster size. Historically these
	// reused min_pts (the ordering's *density* parameter), which over-merges many tight,
	// similar-density clusters at a moderate min_pts (issue #57). min_cluster_size > 0
	// decouples them; 0 => use min_pts, preserving the original behavior (and the pinned
	// chi_test_* results).
	const std::size_t mcs = ( min_cluster_size > 0 ) ? min_cluster_size : min_pts;
	std::vector<std::pair<std::size_t, std::size_t>> clusters;
	std::vector<detail::SDA> SDAs;
	const std::size_t n_reachdists = reach_dists_.size();

	// Unique-point / weighted OPTICS (issue #46): a span's "size" (the min-cluster-size check and
	// the steep-area length cap) must count the ORIGINAL points the span covers, not the number of
	// ordering positions -- otherwise a 2-unique-point steep area standing for 40k pixels is wrongly
	// rejected. cumw is the prefix sum of per-position weights, so the weighted span [a,b) is
	// cumw[b]-cumw[a]. When position_weights is empty every position weighs 1 and cumw[i]==i, so the
	// math below is byte-for-byte the original position arithmetic (keeps chi_test_* pinned).
	std::vector<std::size_t> cumw( n_reachdists + 1 );
	for ( std::size_t i = 0; i < n_reachdists; ++i ) {
		cumw[i + 1] = cumw[i] + ( position_weights.empty() ? std::size_t( 1 ) : position_weights[i] );
	}

	double mib( 0 );
	double max_reach( 0.0 );
	for ( const auto& r : reach_dists_ ) { if ( r.reach_dist > max_reach ) max_reach = r.reach_dist; }


	const auto get_reach_dist = [&reach_dists_, &max_reach]( const std::size_t idx ) -> double {
		assert( idx <= reach_dists_.size() );
		if ( idx == reach_dists_.size() ) return max_reach;
		if ( idx == 0 ) return max_reach;
		const auto r = reach_dists_[idx].reach_dist;
		return ( ( r < 0 ) ? 2 * max_reach : r );
	};
	const auto is_steep_down_pt = [&get_reach_dist, &n_reachdists, &chi]( std::size_t idx ) {
		if ( idx == 0 ) return true;
		if ( idx + 1 >= n_reachdists ) return false;
		return get_reach_dist( idx + 1 ) <= get_reach_dist( idx ) * ( 1 - chi );
	};
	const auto is_steep_up_pt = [&get_reach_dist, &n_reachdists, &chi]( std::size_t idx ) {
		if ( idx + 1 >= n_reachdists ) return true;
		return get_reach_dist( idx + 1 ) * ( 1 - chi ) >= get_reach_dist( idx );
	};
	const auto filter_sdas = [&chi, &steep_area_min_diff, &SDAs, &mib, &get_reach_dist]() {
		std::erase_if( SDAs, [&mib, &chi, &steep_area_min_diff, &get_reach_dist]( const detail::SDA& sda ) -> bool {
			const double f = std::max( chi, steep_area_min_diff );
			return !( mib <= get_reach_dist( sda.begin_idx ) * ( 1 - f ) );
		} );
		for ( auto& sda : SDAs ) {
			sda.mib = std::max( sda.mib, mib );
		}
	};
	const auto get_sda_end = [&n_reachdists, &get_reach_dist, &mcs, &cumw, &is_steep_down_pt]( const std::size_t start_idx ) -> std::size_t {
		assert( is_steep_down_pt( start_idx ) );
		std::size_t last_sd_idx = start_idx;
		std::size_t idx = start_idx + 1;
		while ( idx < n_reachdists ) {
			if ( cumw[idx] - cumw[last_sd_idx] >= mcs ) { return last_sd_idx; }
			if ( get_reach_dist( idx ) > get_reach_dist( idx - 1 ) ) { return last_sd_idx; }
			if ( is_steep_down_pt( idx ) ) { last_sd_idx = idx; }
			idx++;
		}
		return std::max( n_reachdists - 2, last_sd_idx );
	};
	const auto get_sua_end = [&n_reachdists, &get_reach_dist, &mcs, &cumw, &is_steep_up_pt]( const std::size_t start_idx ) -> std::size_t {
		assert( is_steep_up_pt( start_idx ) );
		std::size_t last_su_idx = start_idx;
		std::size_t idx = start_idx + 1;
		while ( idx < n_reachdists ) {
			if ( cumw[idx] - cumw[last_su_idx] >= mcs ) { return last_su_idx; }
			if ( get_reach_dist( idx ) < get_reach_dist( idx - 1 ) ) { return last_su_idx; }
			if ( is_steep_up_pt( idx ) ) { last_su_idx = idx; }
			idx++;
		}
		return std::max( n_reachdists - 2, last_su_idx );
	};
	const auto cluster_borders = [&get_reach_dist, &n_reachdists, &chi]( const detail::SDA& sda, std::size_t sua_begin_idx, std::size_t sua_end_idx ) -> std::pair<std::size_t, std::size_t> {
		double start_reach = get_reach_dist( sda.begin_idx );
		double end_reach = get_reach_dist( std::min( sua_end_idx + 1, n_reachdists - 1 ) );
		if ( detail::in_range( start_reach, end_reach, start_reach * chi ) ) {
			return { sda.begin_idx, sua_end_idx };
		}
		if ( start_reach > end_reach ) {
			std::size_t start_idx = sda.begin_idx + 1;
			while ( start_idx <= sda.end_idx && get_reach_dist( start_idx ) > end_reach ) {
				start_idx++;
			}
			return { start_idx - 1, sua_end_idx };
		}
		if ( start_reach < end_reach ) {
			std::size_t end_idx = sua_end_idx;
			while ( end_idx >= sua_begin_idx && get_reach_dist( end_idx ) >= start_reach ) {
				end_idx--;
			}
			return std::make_pair( sda.begin_idx, end_idx + 1 );
		}
		assert( false );
		return { 0, 0 };
	};
	const auto valid_combination = [&chi, &steep_area_min_diff, &mcs, &cumw, &get_reach_dist]( const detail::SDA& sda, std::size_t sua_begin_idx, std::size_t sua_end_idx ) -> bool {
		const double f = std::max( chi, steep_area_min_diff );
		if ( sda.mib > get_reach_dist( sua_end_idx + 1 ) * ( 1 - f ) ) { return false; }

		std::size_t sda_middle = ( sda.begin_idx + ( sda.end_idx - sda.begin_idx ) / 2 );
		std::size_t sua_middle = ( sua_begin_idx + ( sua_end_idx - sua_begin_idx ) / 2 );
		// Weighted span between the two area midpoints (cumw[sua_middle]-cumw[sda_middle]); equals
		// the original (sua_middle - sda_middle) position distance when unweighted.
		if ( cumw[sua_middle] - cumw[sda_middle] < mcs - 2 ) {
			return false;
		}
		return true;
	};


	for ( std::size_t idx = 0; idx < n_reachdists; idx++ ) {
		double reach_i = get_reach_dist( idx );

		// Start of Steep Down Area?
		if ( idx < n_reachdists && is_steep_down_pt( idx ) ) {
			if ( reach_i > mib ) { mib = reach_i; }
			filter_sdas();
			std::size_t sda_end_idx = get_sda_end( idx );
			if ( reach_i * ( 1.0 - steep_area_min_diff ) < get_reach_dist( sda_end_idx + 1 ) ) {
				continue;
			}
			SDAs.push_back( detail::SDA( idx, sda_end_idx, 0.0 ) );
			idx = sda_end_idx;
			if ( idx < n_reachdists - 1 ) { mib = get_reach_dist( idx + 1 ); }
			continue;
		}
		// Start of Steep Up Area?
		else if ( idx < n_reachdists && is_steep_up_pt( idx ) ) {
			filter_sdas();
			std::size_t sua_end_idx = get_sua_end( idx );
			if ( reach_i > get_reach_dist( sua_end_idx + 1 ) * ( 1.0 - steep_area_min_diff ) ) {
				continue;
			}
			for ( auto& sda : SDAs ) {
				if ( valid_combination( sda, idx, sua_end_idx ) ) {
					clusters.push_back( cluster_borders( sda, idx, sua_end_idx ) );
				}
			}
			idx = sua_end_idx;
			if ( idx < n_reachdists - 1 ) { mib = get_reach_dist( idx + 1 ); }
		} else {
			if ( reach_i > mib ) { mib = reach_i; }
		}
	}
	return clusters;
}


// Arranges flat (begin,end) clusters into a nesting forest of cluster_trees.
inline std::vector<cluster_tree> flat_clusters_to_tree( const std::vector<chi_cluster_indices>& clusters_flat ) {
	// sort clusters_flat such that children are ordered before their parents
	std::vector<std::optional<chi_cluster_indices>> clusters_flat_sorted_m( clusters_flat.size(), std::nullopt );
	std::size_t next_free_idx = 0;
	for ( std::size_t idx = 0; idx < clusters_flat.size(); idx++ ) {
		while ( next_free_idx < clusters_flat_sorted_m.size() && clusters_flat_sorted_m[next_free_idx].has_value() ) {
			next_free_idx++;
		}
		std::size_t idx_pos = next_free_idx;
		std::size_t following_idx = idx + 1;
		while ( following_idx < clusters_flat.size() && clusters_flat[following_idx].second <= clusters_flat[idx].second ) {
			following_idx++;
			idx_pos++;
		}
		clusters_flat_sorted_m[idx_pos] = clusters_flat[idx];
	}

	std::vector<chi_cluster_indices> clusters_flat_sorted;
	clusters_flat_sorted.reserve( clusters_flat.size() );
	for ( const auto& m : clusters_flat_sorted_m ) {
		if ( m.has_value() ) { clusters_flat_sorted.push_back( *m ); }
	}
	assert( clusters_flat_sorted.size() == clusters_flat.size() );

	std::vector<cluster_tree> result;
	std::vector<cluster_tree> cluster_trees;
	cluster_trees.reserve( clusters_flat_sorted.size() );
	for ( const auto& c : clusters_flat_sorted ) {
		cluster_trees.push_back( cluster_tree( c ) );
	}

	auto get_first_parent_idx = [&cluster_trees]( std::size_t idx ) -> std::size_t {
		auto cluster = cluster_trees[idx].get_root().get_data();
		for ( std::size_t first_parent_idx = idx + 1; first_parent_idx < cluster_trees.size(); first_parent_idx++ ) {
			auto parent_cluster = cluster_trees[first_parent_idx].get_root().get_data();
			if ( cluster.first >= parent_cluster.first && cluster.second <= parent_cluster.second ) {
				return first_parent_idx;
			}
		}
		return cluster_trees.size();
	};
	for ( std::size_t idx = 0; idx < cluster_trees.size(); idx++ ) {
		auto first_parent_idx = get_first_parent_idx( idx );
		if ( first_parent_idx >= cluster_trees.size() ) {
			result.push_back( cluster_trees[idx] );
		} else {
			cluster_trees[first_parent_idx].get_root().add_child( cluster_trees[idx].get_root() );
		}
	}
	return result;
}


inline std::vector<cluster_tree> get_chi_clusters( const std::vector<reachability_dist>& reach_dists, const double chi, std::size_t min_pts, const double steep_area_min_diff = 0.0, std::size_t min_cluster_size = 0, const std::vector<std::size_t>& position_weights = {} ) {
	auto clusters_flat = get_chi_clusters_flat( reach_dists, chi, min_pts, steep_area_min_diff, min_cluster_size, position_weights );
	return flat_clusters_to_tree( clusters_flat );
}


inline std::vector<std::vector<std::size_t>> get_cluster_indices(
	const std::vector<reachability_dist>& reach_dists,
	const std::vector<chi_cluster_indices>& clusters ) {
	std::vector<std::vector<std::size_t>> result;
	result.reserve( clusters.size() );
	for ( const auto& c : clusters ) {
		result.push_back( {} );
		result.back().reserve( c.second - c.first + 1 );
		for ( std::size_t idx = c.first; idx <= c.second; idx++ ) {
			const auto& r = reach_dists[idx];
			result.back().push_back( r.point_index );
		}
	}
	return result;
}


template <typename T, std::size_t dimension>
std::vector<std::vector<std::array<T, dimension>>> get_cluster_points(
	const std::vector<reachability_dist>& reach_dists,
	const std::vector<chi_cluster_indices>& clusters,
	const std::vector<std::array<T, dimension>>& points ) {
	const auto clusters_indices = get_cluster_indices( reach_dists, clusters );
	std::vector<std::vector<std::array<T, dimension>>> result;
	result.reserve( clusters_indices.size() );
	for ( const auto& cluster_indices : clusters_indices ) {
		std::vector<std::array<T, dimension>> group;
		group.reserve( cluster_indices.size() );
		for ( const std::size_t idx : cluster_indices ) { group.push_back( points[idx] ); }
		result.push_back( std::move( group ) );
	}
	return result;
}


//=== Xi cluster tree -> point indices =======================================

namespace detail {
// Recursively rebuild a Xi cluster node, replacing its (begin,end) range into the
// ordering with the list of original point indices it covers.
inline Node<std::vector<std::size_t>> chi_node_to_points(
	const Node<chi_cluster_indices>& node, const std::vector<reachability_dist>& reach_dists ) {
	const chi_cluster_indices range = node.get_data();
	std::vector<std::size_t> indices;
	indices.reserve( range.second - range.first + 1 );
	for ( std::size_t idx = range.first; idx <= range.second; ++idx ) {
		indices.push_back( reach_dists[idx].point_index );
	}
	Node<std::vector<std::size_t>> out( std::move( indices ) );
	for ( const auto& child : node.get_children() ) {
		out.add_child( chi_node_to_points( child, reach_dists ) );
	}
	return out;
}
}  // namespace detail

// Map a Xi cluster tree (whose nodes hold (begin,end) ranges into the ordering, as
// produced by get_chi_clusters) onto the same tree shape but with each node holding the
// list of original point indices in its range. The nesting is preserved, so a hierarchical
// result is as easy to consume as the flattened extract_xi -- callers no longer map
// reach[i].point_index by hand. A node's list spans its whole range, so (mirroring
// get_cluster_indices) a parent's list includes the points of its children.
inline Tree<std::vector<std::size_t>> chi_tree_to_points(
	const cluster_tree& tree, const std::vector<reachability_dist>& reach_dists ) {
	return Tree<std::vector<std::size_t>>( detail::chi_node_to_points( tree.get_root(), reach_dists ) );
}

// Forest overload: get_chi_clusters returns a vector of cluster_trees.
inline std::vector<Tree<std::vector<std::size_t>>> chi_tree_to_points(
	const std::vector<cluster_tree>& trees, const std::vector<reachability_dist>& reach_dists ) {
	std::vector<Tree<std::vector<std::size_t>>> result;
	result.reserve( trees.size() );
	for ( const auto& t : trees ) { result.push_back( chi_tree_to_points( t, reach_dists ) ); }
	return result;
}


//=== Convenience entry points ===============================================

// Convert an integer/byte cloud (e.g. uint8 color data) to a float/double cloud,
// since the algorithm requires a floating-point coordinate type:
//   auto cloud = optics::convert_cloud<float>( uint8_points );
template <typename Out, typename In, std::size_t Dim>
std::vector<std::array<Out, Dim>> convert_cloud( const std::vector<std::array<In, Dim>>& in ) {
	std::vector<std::array<Out, Dim>> out;
	out.reserve( in.size() );
	for ( const auto& p : in ) {
		std::array<Out, Dim> q;
		for ( std::size_t k = 0; k < Dim; ++k ) { q[k] = static_cast<Out>( p[k] ); }
		out.push_back( q );
	}
	return out;
}

namespace detail {
// Heuristic flat-cut threshold used when the caller doesn't supply one: a high
// percentile of the finite reachability distances, so only the tallest peaks separate
// clusters. There is no universally-correct flat threshold (that is what the Xi method
// is for) -- look at the reachability plot to tune. Returns 0 if nothing was reached.
inline double default_threshold( const std::vector<reachability_dist>& reach, double percentile = 0.90 ) {
	std::vector<double> finite;
	finite.reserve( reach.size() );
	for ( const auto& r : reach ) { if ( r.reach_dist >= 0.0 ) { finite.push_back( r.reach_dist ); } }
	if ( finite.empty() ) { return 0.0; }
	const std::size_t k = std::min( finite.size() - 1,
		static_cast<std::size_t>( percentile * static_cast<double>( finite.size() ) ) );
	std::nth_element( finite.begin(), finite.begin() + static_cast<std::ptrdiff_t>( k ), finite.end() );
	return finite[k];
}
}  // namespace detail

// One-call flat extraction: compute the OPTICS ordering and cut the reachability plot at
// `threshold`. This is the paper's ExtractDBSCAN -- it yields the *same* clustering DBSCAN
// would at eps = threshold; we do NOT run DBSCAN. Returns one point-index list per cluster
// (unreached/UNDEFINED points become singletons). When `threshold < 0` (the default) an
// educated default is used: a high percentile of the reachabilities (see
// detail::default_threshold) -- no flat threshold is universally right, so inspect the plot.
// `dedup` (default ON, issue #46): collapse bit-identical points to unique weighted points before
// clustering, then expand the result back to ORIGINAL indices. This is the big win for color data
// (a flat region of N identical pixels becomes one weighted point, so the O(neighborhood) cost for
// that region vanishes) and is lossless -- the partition is the same as on the full cloud. Clouds
// with no exact duplicates fall through to the plain unweighted path (byte-for-byte unchanged,
// including the knee auto-epsilon), so dedup only ever helps. Set dedup=false to force the full cloud.
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
std::vector<std::vector<std::size_t>> cluster_threshold(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts, double threshold = -1.0,
	double epsilon = -1.0, NeighborMode mode = NeighborMode::OnDemand, unsigned n_threads = 0,
	bool dedup = true ) {
	if ( dedup ) {
		const auto d = deduplicate( points );
		if ( d.unique_points.size() < points.size() ) {  // actual collapse => weighted OPTICS
			const auto reach = compute_reachability_dists<T, Dim, Backend>(
				d.unique_points, min_pts, epsilon, mode, n_threads, CoreDistMode::Scan, 0, d.weights );
			const double t = ( threshold < 0.0 ) ? detail::default_threshold( reach ) : threshold;
			return expand_clusters_to_original( get_cluster_indices( reach, t ), d.unique_of_original );
		}
		// no duplicates: nothing to gain; use the unweighted path below (preserves knee auto-eps).
	}
	const auto reach = compute_reachability_dists<T, Dim, Backend>( points, min_pts, epsilon, mode, n_threads );
	const double t = ( threshold < 0.0 ) ? detail::default_threshold( reach ) : threshold;
	return get_cluster_indices( reach, t );
}

// Deprecated alias for cluster_threshold: the flat cut is the paper's ExtractDBSCAN, not a
// DBSCAN run. Kept for one version for source compatibility.
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
[[deprecated( "renamed to cluster_threshold (this is OPTICS + a flat cut, not DBSCAN)" )]]
std::vector<std::vector<std::size_t>> cluster_dbscan(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts, double threshold = -1.0,
	double epsilon = -1.0, NeighborMode mode = NeighborMode::OnDemand, unsigned n_threads = 0 ) {
	return cluster_threshold<T, Dim, Backend>( points, min_pts, threshold, epsilon, mode, n_threads );
}

// One-call hierarchical Xi (steep-area) extraction, FLATTENED to a list of clusters:
// compute the ordering and run the Xi method. The nesting is discarded here -- use
// get_chi_clusters(reach, chi, min_pts) for the cluster tree. `chi` defaults to 0.05.
// `dedup` (default ON, issue #46): see cluster_threshold. When duplicates collapse, the Xi steep
// areas are sized by ORIGINAL-point weight (so a few-unique-point but pixel-heavy area is not
// wrongly rejected -- the position_weights argument to get_chi_clusters_flat), and the result is
// expanded back to original indices. No-duplicate clouds use the plain unweighted path unchanged.
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
std::vector<std::vector<std::size_t>> extract_xi(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts, double chi = 0.05,
	double epsilon = -1.0, NeighborMode mode = NeighborMode::OnDemand, unsigned n_threads = 0,
	double steep_area_min_diff = 0.0, std::size_t min_cluster_size = 0, bool dedup = true ) {
	if ( dedup ) {
		const auto d = deduplicate( points );
		if ( d.unique_points.size() < points.size() ) {  // actual collapse => weighted OPTICS + weighted Xi
			const auto reach = compute_reachability_dists<T, Dim, Backend>(
				d.unique_points, min_pts, epsilon, mode, n_threads, CoreDistMode::Scan, 0, d.weights );
			// Per-ordering-position weights: the weight of the unique point sitting at each position.
			std::vector<std::size_t> w_ord( reach.size() );
			for ( std::size_t i = 0; i < reach.size(); ++i ) { w_ord[i] = d.weights[reach[i].point_index]; }
			const auto flat = get_chi_clusters_flat( reach, chi, min_pts, steep_area_min_diff, min_cluster_size, w_ord );
			return expand_clusters_to_original( get_cluster_indices( reach, flat ), d.unique_of_original );
		}
	}
	const auto reach = compute_reachability_dists<T, Dim, Backend>( points, min_pts, epsilon, mode, n_threads );
	const auto flat = get_chi_clusters_flat( reach, chi, min_pts, steep_area_min_diff, min_cluster_size );
	return get_cluster_indices( reach, flat );
}

}  // namespace optics
