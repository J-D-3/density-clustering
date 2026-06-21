// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// sOPTICS vs exact OPTICS: clustering-quality (Rand index) + timing on synthetic,
// L2-normalized blobs (cosine metric, so both algorithms share the same geometry).
//
// What to expect: sOPTICS trades exactness for scalability. Its random-projection
// approximation should PRESERVE the clustering -- a high Rand index (typically ~0.95+)
// against exact OPTICS on the unit sphere. Timing is a tradeoff, not a guaranteed win:
// at low dimension nanoflann's exact kd-tree is extremely fast, so the CEOs projection
// overhead (D * n * Dim) makes sOPTICS slower here; its advantage grows with dimension
// (where exact NN degrades) and dataset scale. This harness makes both visible. The
// paper's order-of-magnitude speedups are vs scikit-learn at millions of points -- see
// the CPU-comparison tracker (#53) for that regime.
//
// Emits CSV to stdout:
//   dataset,n,dim,min_pts,optics_ms,soptics_ms,rand_index
//
// Usage: optics_soptics_compare [scale]   (scale multiplies per-blob counts; default 1).
// Build in a Release config; not a ctest (timings vary by machine).

#include <optics/optics.hpp>
#include <optics/Stopwatch.hpp>
#include <optics/testdata.hpp>

#include "bench_config.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace sw = stopwatch;

namespace {

// Rand index between two flat labelings (fraction of point-pairs that agree on
// same/different cluster). Computed over a strided subsample so the O(k^2) pair loop
// stays bounded while still covering every (contiguous) blob.
double rand_index_sampled( const std::vector<long long>& a, const std::vector<long long>& b, std::size_t max_pts ) {
	const std::size_t n = a.size();
	const std::size_t stride = std::max<std::size_t>( 1, n / std::max<std::size_t>( 1, max_pts ) );
	std::vector<std::size_t> idx;
	for ( std::size_t i = 0; i < n; i += stride ) { idx.push_back( i ); }
	std::size_t agree = 0, total = 0;
	for ( std::size_t i = 0; i < idx.size(); ++i ) {
		for ( std::size_t j = i + 1; j < idx.size(); ++j ) {
			if ( ( a[idx[i]] == a[idx[j]] ) == ( b[idx[i]] == b[idx[j]] ) ) { ++agree; }
			++total;
		}
	}
	return total ? static_cast<double>( agree ) / static_cast<double>( total ) : 1.0;
}

std::vector<long long> labels_from_clusters( std::size_t n, const std::vector<std::vector<std::size_t>>& clusters ) {
	std::vector<long long> labels( n, -1 );
	long long id = 0;
	for ( const auto& c : clusters ) {
		for ( const std::size_t i : c ) { labels[i] = id; }
		++id;
	}
	return labels;
}

template <std::size_t Dim>
void run( std::size_t n_blobs, std::size_t per_blob, std::size_t min_pts, unsigned nt ) {
	auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per_blob, 30.0, 1.0, 321u );
	for ( auto& p : pts ) {  // L2-normalize onto the unit sphere
		double s = 0.0;
		for ( const double c : p ) { s += c * c; }
		s = std::sqrt( s );
		if ( s > 0.0 ) { for ( auto& c : p ) { c /= s; } }
	}
	const std::size_t n = pts.size();
	const double eps = 0.3;

	sw::Stopwatch w1;
	const auto exact = optics::compute_reachability_dists<double, Dim>( pts, min_pts, eps );
	const long long opt_ms = static_cast<long long>( bench::ceil_ms_from_us( w1.elapsed<sw::mus>() ) );

	sw::Stopwatch w2;
	const auto approx = optics::compute_soptics_reachability_dists<double, Dim>(
		pts, min_pts, eps, 1024u, 0u, std::size_t{ 0 }, 7u, nt );
	const long long sopt_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );

	const double thr = 0.5 * eps;
	const auto la = labels_from_clusters( n, optics::get_cluster_indices( exact, thr ) );
	const auto lb = labels_from_clusters( n, optics::get_cluster_indices( approx, thr ) );
	const double ri = rand_index_sampled( la, lb, 3000 );

	std::cout << "blobs," << n << "," << Dim << "," << min_pts << ","
			  << opt_ms << "," << sopt_ms << "," << ri << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }
	const unsigned nt = bench::threads();
	std::cerr << "sOPTICS vs exact OPTICS (threads=" << nt << ", scale=" << scale
			  << "; rand_index ~1.0 == same clustering)\n";
	std::cout << "dataset,n,dim,min_pts,optics_ms,soptics_ms,rand_index\n";
	run<3>( 8, 200 * scale, 8, nt );
	run<3>( 12, 350 * scale, 10, nt );
	run<16>( 6, 200 * scale, 8, nt );
	return 0;
}
