// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Optional Boost.Geometry R*-tree neighbor-search backend. Only compiled when
// OPTICS_ENABLE_BOOST_RTREE is defined (see CMake option). The core never
// includes Boost otherwise.

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace optics {

namespace detail {

namespace bg = boost::geometry;

template <typename BoostPoint, typename T, std::size_t Dim, std::size_t... I>
inline void assign_boost_coords( BoostPoint& bp, const std::array<T, Dim>& a, std::index_sequence<I...> ) {
	( bg::set<I>( bp, a[I] ), ... );
}

template <typename BoostPoint, typename T, std::size_t Dim>
inline BoostPoint to_boost_point( const std::array<T, Dim>& a ) {
	BoostPoint bp;
	assign_boost_coords( bp, a, std::make_index_sequence<Dim>{} );
	return bp;
}

}  // namespace detail

// Neighbor-search backend backed by a boost::geometry R*-tree. Satisfies the
// NeighborSearch concept. Points are converted into the tree once at construction.
template <typename T, std::size_t Dim>
class BoostRTreeBackend {
public:
	using Point = std::array<T, Dim>;

	explicit BoostRTreeBackend( const std::vector<Point>& points ) {
		std::vector<Value> values;
		values.reserve( points.size() );
		for ( std::size_t i = 0; i < points.size(); ++i ) {
			values.emplace_back( detail::to_boost_point<BPoint>( points[i] ), i );
		}
		rtree_ = RTree( values.begin(), values.end() );  // bulk-load (packing) constructor
	}

	// Append the indices of all points within Euclidean distance r of p to out.
	void radius_search( const Point& p, T r, std::vector<std::size_t>& out ) const {
		const BPoint query = detail::to_boost_point<BPoint>( p );

		Point lo, hi;
		for ( std::size_t i = 0; i < Dim; ++i ) {
			lo[i] = static_cast<T>( p[i] - r );
			hi[i] = static_cast<T>( p[i] + r );
		}
		const Box box( detail::to_boost_point<BPoint>( lo ), detail::to_boost_point<BPoint>( hi ) );

		std::vector<Value> hits;
		rtree_.query( boost::geometry::index::intersects( box ), std::back_inserter( hits ) );
		out.reserve( out.size() + hits.size() );
		for ( const auto& h : hits ) {
			// The box query is a Chebyshev pre-filter; keep only true Euclidean neighbors.
			if ( boost::geometry::distance( h.first, query ) <= r ) { out.push_back( h.second ); }
		}
	}

private:
	using BPoint = boost::geometry::model::point<T, Dim, boost::geometry::cs::cartesian>;
	using Value = std::pair<BPoint, std::size_t>;
	using Box = boost::geometry::model::box<BPoint>;
	using RTree = boost::geometry::index::rtree<Value, boost::geometry::index::rstar<16>>;

	RTree rtree_;
};

}  // namespace optics
