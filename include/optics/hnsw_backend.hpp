// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Optional HNSW (Hierarchical Navigable Small World) approximate neighbor-search backend,
// backed by the vendored header-only hnswlib (include/optics/hnswlib/, Apache-2.0). Compiled
// when OPTICS_ENABLE_HNSW is defined (see the CMake option); the core never includes
// hnswlib otherwise.
//
// Why: in high dimensions (e.g. the 16-D perspective-transform regime) the exact KD-tree
// degrades toward a linear scan, and even nanoflann's eps-approximate mode has little to
// prune. HNSW is a graph index whose query cost is largely dimension-independent; it trades
// a bounded recall loss for a large speedup. Recall depends on the build parameters
// (M, ef_construction) and the search breadth (ef). Coordinates are stored as float (hnswlib's
// L2 space is float), an additional approximation on top of the graph's.
//
// Models the NeighborSearch concept (radius_search) and the optional KnnCoreDist capability
// (knn_core_dist) -- the latter is HNSW's natural strength and is what CoreDistMode::Knn and
// the knee epsilon estimator use.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

// hnswlib is vendored third-party; silence its warnings (intrinsics, unused statics, etc.)
// so the project can build warnings-as-errors without modifying the vendored headers.
#if defined(_MSC_VER)
#  pragma warning(push, 0)
#endif
#include "hnswlib/hnswlib.h"
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

namespace optics {

// Approximate HNSW backend. M and ef_construction govern index quality/build cost; ef_search
// governs query recall/cost (0 => a sensible default derived from ef_construction). hnswlib's
// L2 space works in SQUARED distance, like nanoflann, so the radius is squared internally.
template <typename T, std::size_t Dim>
class HnswBackend {
public:
	using Point = std::array<T, Dim>;

	explicit HnswBackend( const std::vector<Point>& points, std::size_t M = 16,
						   std::size_t ef_construction = 200, std::size_t ef_search = 0 )
		: space_( Dim ), n_( points.size() ),
		  ef_search_( ef_search != 0 ? ef_search : std::max<std::size_t>( ef_construction, 64 ) ) {
		index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
			&space_, std::max<std::size_t>( n_, 1 ), M, ef_construction );
		index_->setEf( ef_search_ );
		std::array<float, Dim> buf;
		for ( std::size_t i = 0; i < n_; ++i ) {
			for ( std::size_t c = 0; c < Dim; ++c ) { buf[c] = static_cast<float>( points[i][c] ); }
			index_->addPoint( buf.data(), static_cast<hnswlib::labeltype>( i ) );
		}
	}

	// Append the indices of all points within (approximate) Euclidean distance r of p to out.
	// Uses hnswlib's native range search: a SINGLE graph traversal driven by an epsilon stop
	// condition collects every point within r (filtered to <= r internally, so no spurious
	// neighbors and no per-query growth). min_candidates bounds recall (explore at least that
	// many before stopping at the r frontier); max_candidates caps a very dense neighborhood.
	void radius_search( const Point& p, T r, std::vector<std::size_t>& out ) const {
		if ( n_ == 0 ) { return; }
		const std::array<float, Dim> q = to_float( p );
		const float r_sq = static_cast<float>( r ) * static_cast<float>( r );  // L2Space works in squared dist
		const std::size_t min_c = std::min<std::size_t>( n_, ef_search_ );
		hnswlib::EpsilonSearchStopCondition<float> stop( r_sq, min_c, n_ );
		auto res = index_->searchStopConditionClosest( q.data(), stop );  // all within r_sq, closer-first
		out.reserve( out.size() + res.size() );
		for ( const auto& dl : res ) { out.push_back( static_cast<std::size_t>( dl.second ) ); }
	}

	// Core-distance via a k-NN query: (approximate) distance to the min_pts-th nearest point,
	// or nullopt if that neighbor lies beyond r (or fewer than min_pts points exist). HNSW's
	// native operation; powers CoreDistMode::Knn and the knee epsilon estimator.
	std::optional<double> knn_core_dist( const Point& p, std::size_t min_pts, T r ) const {
		if ( n_ < min_pts ) { return std::nullopt; }
		const std::array<float, Dim> q = to_float( p );
		auto res = index_->searchKnnCloserFirst( q.data(), min_pts );
		if ( res.size() < min_pts ) { return std::nullopt; }
		const double kth_sq = static_cast<double>( res[min_pts - 1].first );
		if ( kth_sq > static_cast<double>( r ) * static_cast<double>( r ) ) { return std::nullopt; }
		return std::sqrt( kth_sq );
	}

private:
	static std::array<float, Dim> to_float( const Point& p ) {
		std::array<float, Dim> q;
		for ( std::size_t c = 0; c < Dim; ++c ) { q[c] = static_cast<float>( p[c] ); }
		return q;
	}

	hnswlib::L2Space space_;  // declared before index_: constructed first, destroyed last
	std::size_t n_;
	std::size_t ef_search_;
	std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

}  // namespace optics
