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
//
// Two projection backends (CeosParams::projection):
//   - Gaussian (default): D explicit N(0,1) vectors; each projection is an O(Dim) dot product,
//     computed on the fly so no n x D matrix is materialized (O(D*m) memory).
//   - Structured (opt-in, issue #58): random "spinners" x -> H D3 H D2 H D1 x (sign-flip then
//     fast Walsh-Hadamard transform, three rounds) approximate Gaussian projections at
//     O(D log Dim) per point instead of O(D*Dim). The whole n x D projection table is
//     materialized once (O(n*D) memory) because the FHT produces all of a block's outputs
//     together. This is a HIGH-DIMENSION optimization: measured ~1.2-1.4x faster at >= 64-D with
//     unchanged recall, but break-even at ~16-D and recall-lossy at very low Dim (the Hadamard
//     block next_pow2(Dim) is tiny there), so the default stays Gaussian. See perf/README.md.

#include "hadamard.hpp"
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
	enum class Projection { Gaussian, Structured };
	unsigned n_projections = 1024;   // D: number of random projection vectors
	unsigned k = 0;                  // top-k closest/furthest vectors per point (0 => default 10)
	std::size_t m = 0;               // top-m closest/furthest points per vector (0 => 2*min_pts)
	unsigned seed = 42;
	unsigned n_threads = 0;          // 0 => hardware concurrency
	Projection projection = Projection::Gaussian;  // Structured = FHT spinners (issue #58)
};

// Build approximate epsilon-neighborhoods via CEOs. `points` should be L2-normalized
// by the caller (cosine metric); distances use the Euclidean detail::square_dist,
// which is monotonic in cosine distance on the unit sphere, so the resulting OPTICS
// ordering matches the cosine one. Returns, for each point, the indices of its
// approximate neighbors within `eps` -- including the point itself, matching the
// self-match a radius search returns. The relation is symmetrized: q in N(x) iff
// x in N(q).
// When out_sq != nullptr it is filled with each neighbor's SQUARED distance, parallel to
// the returned neighbor lists (and reusing the detail::square_dist already computed in the
// eps-filter, so the caller need not recompute it for the core-distance / relaxation --
// the sOPTICS analogue of the RadiusSearchWithDists fast path, issue #55). Because every
// distance comes from detail::square_dist (accumulated in double regardless of T), this is
// bit-identical for both float and double sOPTICS -- no double-only gate is needed.
template <class T, std::size_t Dim>
std::vector<std::vector<std::size_t>> ceos_neighbors(
		const std::vector<std::array<T, Dim>>& points, double eps,
		std::size_t min_pts, const CeosParams& params = {},
		std::vector<std::vector<double>>* out_sq = nullptr ) {
	const std::size_t n = points.size();
	std::vector<std::vector<std::size_t>> neighbors( n );
	if ( out_sq ) { out_sq->assign( n, {} ); }
	if ( n == 0 ) { return neighbors; }

	const unsigned D = std::max( 1u, params.n_projections );
	const unsigned k = std::min<unsigned>( D, params.k ? params.k : 10u );
	const std::size_t m = std::min<std::size_t>(
		n, params.m ? params.m : std::max<std::size_t>( 2 * min_pts, 1 ) );
	const double eps_sq = ( eps <= 0.0 ) ? 0.0 : eps * eps;

	const bool structured = ( params.projection == CeosParams::Projection::Structured );

	// 1. Build the projection backend.
	// Gaussian: D explicit N(0,1) vectors, dotted on the fly (no n x D matrix).
	// Structured: precompute the n x D projection table via FHT spinners (see header).
	std::mt19937 gen( params.seed );
	std::normal_distribution<double> gauss( 0.0, 1.0 );
	std::vector<std::array<double, Dim>> R;  // Gaussian path only
	std::vector<double> proj;                // Structured path only: flat n x D, row-major

	if ( structured ) {
		const std::size_t d0 = detail::next_pow2( Dim );             // padded, power-of-two block width
		const std::size_t nblocks = ( D + d0 - 1 ) / d0;            // stacked blocks to reach >= D outputs
		// Three random sign rows per block (the D1/D2/D3 diagonals), generated up front so the
		// result is deterministic in seed and independent of the thread count.
		std::vector<std::vector<double>> signs( nblocks * 3, std::vector<double>( d0 ) );
		std::uniform_int_distribution<int> coin( 0, 1 );
		for ( auto& row : signs ) {
			for ( std::size_t t = 0; t < d0; ++t ) { row[t] = coin( gen ) ? 1.0 : -1.0; }
		}
		proj.assign( n * static_cast<std::size_t>( D ), 0.0 );
		parallel_for( params.n_threads, n, [&]( std::size_t i ) {
			std::vector<double> x( d0 );
			for ( std::size_t b = 0; b < nblocks; ++b ) {
				for ( std::size_t t = 0; t < d0; ++t ) { x[t] = ( t < Dim ) ? static_cast<double>( points[i][t] ) : 0.0; }
				for ( std::size_t r = 0; r < 3; ++r ) {
					const auto& s = signs[b * 3 + r];
					for ( std::size_t t = 0; t < d0; ++t ) { x[t] *= s[t]; }
					detail::fwht_inplace( x );
				}
				const std::size_t base = b * d0;
				for ( std::size_t t = 0; t < d0 && base + t < D; ++t ) { proj[i * D + base + t] = x[t]; }
			}
		} );
	} else {
		R.resize( D );
		for ( unsigned j = 0; j < D; ++j ) {
			for ( std::size_t c = 0; c < Dim; ++c ) { R[j][c] = gauss( gen ); }
		}
	}

	const auto project = [&]( std::size_t i, unsigned j ) -> double {
		if ( structured ) { return proj[i * D + j]; }
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
	// Carry each candidate's squared distance alongside its index so it can be reused
	// downstream (out_sq). dist(q,x) == dist(x,q), so a reverse edge reuses the same value.
	using Cand = std::pair<std::size_t, double>;  // (neighbor index, squared distance)
	const auto by_index = []( const Cand& a, const Cand& b ) { return a.first < b.first; };
	const auto same_index = []( const Cand& a, const Cand& b ) { return a.first == b.first; };

	std::vector<std::vector<Cand>> directed( n );
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
		out.emplace_back( q, 0.0 );  // self-match (distance 0)
		for ( unsigned vi = 0; vi < qvecs.size(); ++vi ) {
			// closest vectors contribute their closest points; furthest vectors their
			// furthest points (both directions enrich the neighborhood, per the paper).
			const auto& pts = ( vi < kk ) ? close_pts[qvecs[vi]] : far_pts[qvecs[vi]];
			for ( const std::size_t x : pts ) {
				if ( x == q ) { continue; }
				const double sq = detail::square_dist( points[q], points[x] );
				if ( sq <= eps_sq ) { out.emplace_back( x, sq ); }
			}
		}
		std::sort( out.begin(), out.end(), by_index );
		out.erase( std::unique( out.begin(), out.end(), same_index ), out.end() );
	} );

	// 4. Symmetrize from the directed lists: x in N(q) <=> q in N(x), carrying the (shared)
	//    squared distance. The reverse-edge pass writes across points, so it is
	//    single-threaded; the dedup + split into indices (+ optional out_sq) is parallel.
	std::vector<std::vector<Cand>> sym( n );
	for ( std::size_t q = 0; q < n; ++q ) {
		for ( const Cand& c : directed[q] ) {
			sym[q].push_back( c );
			if ( c.first != q ) { sym[c.first].emplace_back( q, c.second ); }
		}
	}
	parallel_for( params.n_threads, n, [&]( std::size_t q ) {
		auto& v = sym[q];
		std::sort( v.begin(), v.end(), by_index );
		v.erase( std::unique( v.begin(), v.end(), same_index ), v.end() );
		neighbors[q].reserve( v.size() );
		for ( const Cand& c : v ) { neighbors[q].push_back( c.first ); }
		if ( out_sq ) {
			( *out_sq )[q].reserve( v.size() );
			for ( const Cand& c : v ) { ( *out_sq )[q].push_back( c.second ); }
		}
	} );

	return neighbors;
}

}  // namespace optics::detail
