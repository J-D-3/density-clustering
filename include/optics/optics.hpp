// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

#include "backend.hpp"
#include "tree.hpp"
#include "version.hpp"
#include "detail/math.hpp"
#include "detail/thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
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

	const double d = static_cast<double>( dimension );
	const auto space = detail::bounding_box( points );
	const double space_volume = detail::hypercuboid_volume( space.first, space.second );

	const double space_per_minpts_points = ( space_volume / static_cast<double>( points.size() ) ) * static_cast<double>( min_pts );
	const double n_dim_unit_ball_vol = std::sqrt( std::pow( detail::pi, d ) ) / std::tgamma( d / 2.0 + 1.0 );
	return std::pow( space_per_minpts_points / n_dim_unit_ball_vol, 1.0 / d );
}


//=== The OPTICS ordering ====================================================

// Computes the OPTICS cluster-ordering and reachability distances.
//   T, Dim   : coordinate type (float/double) and dimensionality (deduced from points).
//   Backend  : neighbor-search backend; defaults to nanoflann.
//   epsilon  : generating distance; auto-estimated when <= 0.
//   mode     : Precompute (parallel, default) or OnDemand neighbor acquisition.
//   n_threads: worker threads for Precompute (0 => hardware concurrency).
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
std::vector<reachability_dist> compute_reachability_dists(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts,
	double epsilon = -1.0, NeighborMode mode = NeighborMode::Precompute, unsigned n_threads = 0 ) {

	static_assert( std::is_floating_point_v<T>, "compute_reachability_dists: coordinate type 'T' must be float or double" );
	static_assert( Dim >= 1, "compute_reachability_dists: dimension must be >= 1" );
	static_assert( NeighborSearch<Backend, T, Dim>, "Backend does not satisfy the NeighborSearch concept" );

	if ( min_pts < 1 ) { throw std::invalid_argument( "compute_reachability_dists: min_pts must be >= 1" ); }
	if ( points.empty() ) { return {}; }

	double eps = epsilon;
	if ( eps <= 0.0 ) { eps = epsilon_estimation( points, min_pts ); }
	if ( eps <= 0.0 ) { eps = 1.0; }  // degenerate input (all points identical): any positive radius works
	const T eps_t = static_cast<T>( eps );

	const Backend backend( points );
	const std::size_t n = points.size();

	std::vector<char> processed( n, 0 );
	std::vector<std::size_t> ordered_list;
	ordered_list.reserve( n );
	std::vector<double> reachability( n, -1.0 );

	// Neighbor acquisition: precompute-all (parallel) or on-demand.
	std::vector<std::vector<std::size_t>> neighbors;
	if ( mode == NeighborMode::Precompute ) {
		neighbors.resize( n );
		detail::parallel_for( n_threads, n, [&]( std::size_t i ) {
			backend.radius_search( points[i], eps_t, neighbors[i] );
		} );
	}
	std::vector<std::size_t> ondemand_buf;
	const auto neighbors_of = [&]( std::size_t idx ) -> const std::vector<std::size_t>& {
		if ( mode == NeighborMode::Precompute ) { return neighbors[idx]; }
		ondemand_buf.clear();
		backend.radius_search( points[idx], eps_t, ondemand_buf );
		return ondemand_buf;
	};

	// Reused across points; drained to empty whenever an expansion finishes.
	detail::seed_queue seeds;
	const auto relax_neighbors = [&]( std::size_t idx, const std::vector<std::size_t>& nbrs, double core_dist ) {
		for ( const std::size_t o : nbrs ) {
			if ( processed[o] ) { continue; }
			const double new_rd = std::max( core_dist, detail::dist( points[idx], points[o] ) );
			if ( reachability[o] < 0.0 || new_rd < reachability[o] ) {
				reachability[o] = new_rd;  // authoritative; the prior heap entry becomes stale
				seeds.push( reachability_dist( o, new_rd ) );
			}
		}
	};

	for ( std::size_t point_idx = 0; point_idx < n; point_idx++ ) {
		if ( processed[point_idx] ) { continue; }
		processed[point_idx] = 1;
		ordered_list.push_back( point_idx );

		const auto& neighbor_indices = neighbors_of( point_idx );
		const auto core_dist = detail::compute_core_dist( points[point_idx], points, neighbor_indices, min_pts );
		if ( core_dist.has_value() ) { relax_neighbors( point_idx, neighbor_indices, *core_dist ); }

		while ( !seeds.empty() ) {
			const reachability_dist s = seeds.top();
			seeds.pop();
			// Lazy deletion: skip entries already processed or superseded by a smaller push.
			if ( processed[s.point_index] || s.reach_dist != reachability[s.point_index] ) { continue; }
			processed[s.point_index] = 1;
			ordered_list.push_back( s.point_index );

			const auto& s_neighbor_indices = neighbors_of( s.point_index );
			const auto s_core_dist = detail::compute_core_dist( points[s.point_index], points, s_neighbor_indices, min_pts );
			if ( s_core_dist.has_value() ) { relax_neighbors( s.point_index, s_neighbor_indices, *s_core_dist ); }
		}
	}
	assert( ordered_list.size() == n );

	std::vector<reachability_dist> result;
	result.reserve( n );
	for ( const std::size_t point_idx : ordered_list ) {
		result.emplace_back( point_idx, reachability[point_idx] );
	}
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


inline std::vector<chi_cluster_indices> get_chi_clusters_flat( const std::vector<reachability_dist>& reach_dists_, const double chi, std::size_t min_pts, double steep_area_min_diff = 0.0 ) {
	std::vector<std::pair<std::size_t, std::size_t>> clusters;
	std::vector<detail::SDA> SDAs;
	const std::size_t n_reachdists = reach_dists_.size();
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
	const auto get_sda_end = [&n_reachdists, &get_reach_dist, &min_pts, &is_steep_down_pt]( const std::size_t start_idx ) -> std::size_t {
		assert( is_steep_down_pt( start_idx ) );
		std::size_t last_sd_idx = start_idx;
		std::size_t idx = start_idx + 1;
		while ( idx < n_reachdists ) {
			if ( idx - last_sd_idx >= min_pts ) { return last_sd_idx; }
			if ( get_reach_dist( idx ) > get_reach_dist( idx - 1 ) ) { return last_sd_idx; }
			if ( is_steep_down_pt( idx ) ) { last_sd_idx = idx; }
			idx++;
		}
		return std::max( n_reachdists - 2, last_sd_idx );
	};
	const auto get_sua_end = [&n_reachdists, &get_reach_dist, &min_pts, &is_steep_up_pt]( const std::size_t start_idx ) -> std::size_t {
		assert( is_steep_up_pt( start_idx ) );
		std::size_t last_su_idx = start_idx;
		std::size_t idx = start_idx + 1;
		while ( idx < n_reachdists ) {
			if ( idx - last_su_idx >= min_pts ) { return last_su_idx; }
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
	const auto valid_combination = [&chi, &steep_area_min_diff, &min_pts, &get_reach_dist]( const detail::SDA& sda, std::size_t sua_begin_idx, std::size_t sua_end_idx ) -> bool {
		const double f = std::max( chi, steep_area_min_diff );
		if ( sda.mib > get_reach_dist( sua_end_idx + 1 ) * ( 1 - f ) ) { return false; }

		std::size_t sda_middle = ( sda.begin_idx + ( sda.end_idx - sda.begin_idx ) / 2 );
		std::size_t sua_middle = ( sua_begin_idx + ( sua_end_idx - sua_begin_idx ) / 2 );
		if ( sua_middle - sda_middle < min_pts - 2 ) {
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


inline std::vector<cluster_tree> get_chi_clusters( const std::vector<reachability_dist>& reach_dists, const double chi, std::size_t min_pts, const double steep_area_min_diff = 0.0 ) {
	auto clusters_flat = get_chi_clusters_flat( reach_dists, chi, min_pts, steep_area_min_diff );
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

// One-call DBSCAN-style clustering: compute the ordering and cut at a threshold.
// Returns one index list per cluster (noise points become singletons).
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
std::vector<std::vector<std::size_t>> cluster_dbscan(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts, double threshold,
	double epsilon = -1.0, NeighborMode mode = NeighborMode::Precompute, unsigned n_threads = 0 ) {
	const auto reach = compute_reachability_dists<T, Dim, Backend>( points, min_pts, epsilon, mode, n_threads );
	return get_cluster_indices( reach, threshold );
}

// Xi (steep-area) clusters as point-index lists, in one call from a cluster-ordering.
inline std::vector<std::vector<std::size_t>> extract_xi(
	const std::vector<reachability_dist>& reach_dists, double chi, std::size_t min_pts,
	double steep_area_min_diff = 0.0 ) {
	const auto flat = get_chi_clusters_flat( reach_dists, chi, min_pts, steep_area_min_diff );
	return get_cluster_indices( reach_dists, flat );
}

}  // namespace optics
