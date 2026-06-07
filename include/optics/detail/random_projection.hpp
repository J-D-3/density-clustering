// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// CEOs random-projection neighbor index for sOPTICS (scalable OPTICS).
//
// Implements the candidate-neighbor discovery of sDBSCAN/sOPTICS (Xu & Pham,
// "Scalable Density-based Clustering with Random Projections", NeurIPS 2024; the
// paper is archived under documentation/). The trick is CEOs -- Concomitants of
// Extreme Order Statistics: project every point onto D Gaussian random vectors;
// for a point q, the random vectors onto which q has the most extreme (largest or
// smallest) projection preserve inner-product order with high probability, so the
// points ranked most extreme on those few vectors are q's approximate nearest
// neighbors. We gather candidates from q's top-k extreme vectors (the top-m extreme
// points of each), keep those within epsilon, and symmetrize. This finds approximate
// eps-neighborhoods in ~O(Dim * n * k * m) instead of the O(n^2) of exact all-pairs.
//
// Reimplemented from the paper only: the reference repo NinhPham/sDbscan is
// unlicensed, so no third-party code is used and the library stays dependency-free.
// A structured (Hadamard/FHT) projection is a future speedup; this first version uses
// plain Gaussian projections.

#include "math.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace optics::detail {

// Tunables for the CEOs index. Defaults follow the paper (D = 1024) and scale m with
// min_pts; k is a small constant (more extreme vectors => higher recall, but more
// distance computations per point: the candidate pool is at most 2 * k * m).
struct CeosParams {
	unsigned n_projections = 1024;   // D: number of Gaussian random vectors
	unsigned k = 0;                  // top-k closest/furthest vectors per point (0 => default 10)
	std::size_t m = 0;               // top-m closest/furthest points per vector (0 => 2*min_pts)
	unsigned seed = 42;
	unsigned n_threads = 0;          // 0 => hardware concurrency
};

// Build approximate epsilon-neighborhoods via CEOs. `points` should be L2-normalized
// by the caller (cosine metric); distances use the Euclidean detail::square_dist,
// which is monotonic in cosine distance on the unit sphere, so the resulting OPTICS
// ordering matches the cosine one. Returns, for each point, the indices of its
// approximate neighbors within `eps` -- including the point itself, matching the
// self-match a radius search returns. The relation is symmetrized: q in N(x) iff
// x in N(q).
template <class T, std::size_t Dim>
std::vector<std::vector<std::size_t>> ceos_neighbors(
		const std::vector<std::array<T, Dim>>& points, double eps,
		std::size_t min_pts, const CeosParams& params = {} ) {
	const std::size_t n = points.size();
	std::vector<std::vector<std::size_t>> neighbors( n );
	if ( n == 0 ) { return neighbors; }

	const unsigned D = std::max( 1u, params.n_projections );
	const unsigned k = std::min<unsigned>( D, params.k ? params.k : 10u );
	const std::size_t m = std::min<std::size_t>(
		n, params.m ? params.m : std::max<std::size_t>( 2 * min_pts, 1 ) );
	const double eps_sq = ( eps <= 0.0 ) ? 0.0 : eps * eps;

	// 1. D Gaussian random projection vectors r_j in R^Dim (coordinates ~ N(0,1), as
	//    in the paper -- not unit-normalized).
	std::mt19937 gen( params.seed );
	std::normal_distribution<double> gauss( 0.0, 1.0 );
	std::vector<std::array<double, Dim>> R( D );
	for ( unsigned j = 0; j < D; ++j ) {
		for ( std::size_t c = 0; c < Dim; ++c ) { R[j][c] = gauss( gen ); }
	}

	const auto project = [&]( std::size_t i, unsigned j ) -> double {
		double s = 0.0;
		for ( std::size_t c = 0; c < Dim; ++c ) { s += static_cast<double>( points[i][c] ) * R[j][c]; }
		return s;
	};

	// 2a. Per vector r_j: its top-m closest (largest projection) and top-m furthest
	//     (smallest projection) points. Parallel over vectors -- each is independent,
	//     so no shared-state contention; memory is O(D*m), not the O(n*D) of a full
	//     projection matrix.
	std::vector<std::vector<std::size_t>> close_pts( D ), far_pts( D );
	const std::size_t mm = std::min<std::size_t>( m, n );
	parallel_for( params.n_threads, D, [&]( std::size_t jj ) {
		const unsigned j = static_cast<unsigned>( jj );
		std::vector<std::pair<double, std::size_t>> col( n );
		for ( std::size_t i = 0; i < n; ++i ) { col[i] = { project( i, j ), i }; }
		std::partial_sort( col.begin(), col.begin() + mm, col.end(),
			[]( const auto& a, const auto& b ) { return a.first > b.first; } );
		close_pts[j].reserve( mm );
		for ( std::size_t t = 0; t < mm; ++t ) { close_pts[j].push_back( col[t].second ); }
		std::partial_sort( col.begin(), col.begin() + mm, col.end(),
			[]( const auto& a, const auto& b ) { return a.first < b.first; } );
		far_pts[j].reserve( mm );
		for ( std::size_t t = 0; t < mm; ++t ) { far_pts[j].push_back( col[t].second ); }
	} );

	// 2b + 3. Per point q: find its top-k closest/furthest vectors, gather candidates
	//     from those vectors' extreme points, keep candidates within eps. Parallel over
	//     points -- each writes only its own directed list. Self is added (dist 0),
	//     matching radius-search self-matches.
	std::vector<std::vector<std::size_t>> directed( n );
	parallel_for( params.n_threads, n, [&]( std::size_t q ) {
		std::vector<std::pair<double, unsigned>> row( D );
		for ( unsigned j = 0; j < D; ++j ) { row[j] = { project( q, j ), j }; }
		const unsigned kk = std::min<unsigned>( k, D );
		std::vector<unsigned> qvecs;
		qvecs.reserve( 2u * kk );
		std::partial_sort( row.begin(), row.begin() + kk, row.end(),
			[]( const auto& a, const auto& b ) { return a.first > b.first; } );  // q's closest vectors
		for ( unsigned t = 0; t < kk; ++t ) { qvecs.push_back( row[t].second ); }
		std::partial_sort( row.begin(), row.begin() + kk, row.end(),
			[]( const auto& a, const auto& b ) { return a.first < b.first; } );  // q's furthest vectors
		for ( unsigned t = 0; t < kk; ++t ) { qvecs.push_back( row[t].second ); }

		auto& out = directed[q];
		out.push_back( q );
		for ( unsigned vi = 0; vi < qvecs.size(); ++vi ) {
			// closest vectors contribute their closest points; furthest vectors their
			// furthest points (both directions enrich the neighborhood, per the paper).
			const auto& pts = ( vi < kk ) ? close_pts[qvecs[vi]] : far_pts[qvecs[vi]];
			for ( const std::size_t x : pts ) {
				if ( x == q ) { continue; }
				if ( detail::square_dist( points[q], points[x] ) <= eps_sq ) { out.push_back( x ); }
			}
		}
		std::sort( out.begin(), out.end() );
		out.erase( std::unique( out.begin(), out.end() ), out.end() );
	} );

	// 4. Symmetrize from the directed lists: x in N(q) <=> q in N(x). The reverse-edge
	//    pass writes across points, so it is single-threaded; the dedup is parallel.
	for ( std::size_t q = 0; q < n; ++q ) {
		for ( const std::size_t x : directed[q] ) {
			neighbors[q].push_back( x );
			if ( x != q ) { neighbors[x].push_back( q ); }
		}
	}
	parallel_for( params.n_threads, n, [&]( std::size_t q ) {
		auto& v = neighbors[q];
		std::sort( v.begin(), v.end() );
		v.erase( std::unique( v.begin(), v.end() ), v.end() );
	} );

	return neighbors;
}

}  // namespace optics::detail
