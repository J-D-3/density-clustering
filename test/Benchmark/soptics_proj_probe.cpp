// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Gaussian vs structured (FHT spinner) CEOs projections for sOPTICS (issue #58), across dimension.
// The CEOs projection is sOPTICS's dominant small-scale cost. Gaussian projection is O(D*Dim) per
// point; the structured spinner (x -> H D3 H D2 H D1 x) is O(D log Dim), so its time advantage grows
// with dimension. This harness shows both the projection-time crossover and that the structured
// path preserves the clustering (Rand index vs exact OPTICS stays comparable to Gaussian).
//
// Emits CSV to stdout:
//   dim,n,min_pts,optics_ms,gauss_ms,gauss_rand,struct_ms,struct_rand
// Build in a Release config; not a ctest (timings vary by machine).
//
// Usage: optics_soptics_proj_probe [scale]   (scale multiplies per-blob counts; default 1).

#include <optics/optics.hpp>
#include <optics/Stopwatch.hpp>
#include <optics/testdata.hpp>

#include "bench_config.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace sw = stopwatch;

namespace {

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
	for ( const auto& c : clusters ) { for ( const std::size_t i : c ) { labels[i] = id; } ++id; }
	return labels;
}

template <std::size_t Dim>
void run( std::size_t n_blobs, std::size_t per_blob, std::size_t min_pts, unsigned nt ) {
	auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per_blob, 30.0, 1.0, 321u );
	for ( auto& p : pts ) {  // L2-normalize onto the unit sphere (cosine metric)
		double s = 0.0;
		for ( const double c : p ) { s += c * c; }
		s = std::sqrt( s );
		if ( s > 0.0 ) { for ( auto& c : p ) { c /= s; } }
	}
	const std::size_t n = pts.size();
	const double eps = 0.3, thr = 0.5 * eps;

	sw::Stopwatch w0;
	const auto exact = optics::compute_reachability_dists<double, Dim>( pts, min_pts, eps );
	const long long opt_ms = static_cast<long long>( bench::ceil_ms_from_us( w0.elapsed<sw::mus>() ) );
	const auto le = labels_from_clusters( n, optics::get_cluster_indices( exact, thr ) );

	sw::Stopwatch w1;
	const auto g = optics::compute_soptics_reachability_dists<double, Dim>(
		pts, min_pts, eps, 1024u, 0u, std::size_t{ 0 }, 7u, nt, optics::Metric::Cosine, 0.0,
		std::vector<std::size_t>{}, optics::SopticsProjection::Gaussian );
	const long long g_ms = static_cast<long long>( bench::ceil_ms_from_us( w1.elapsed<sw::mus>() ) );
	const double g_rand = rand_index_sampled( le, labels_from_clusters( n, optics::get_cluster_indices( g, thr ) ), 3000 );

	sw::Stopwatch w2;
	const auto s = optics::compute_soptics_reachability_dists<double, Dim>(
		pts, min_pts, eps, 1024u, 0u, std::size_t{ 0 }, 7u, nt, optics::Metric::Cosine, 0.0,
		std::vector<std::size_t>{}, optics::SopticsProjection::Structured );
	const long long s_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );
	const double s_rand = rand_index_sampled( le, labels_from_clusters( n, optics::get_cluster_indices( s, thr ) ), 3000 );

	std::cout << Dim << "," << n << "," << min_pts << "," << opt_ms << "," << g_ms << "," << g_rand
			  << "," << s_ms << "," << s_rand << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }
	const unsigned nt = bench::threads();
	std::cerr << "sOPTICS Gaussian vs structured projections (threads=" << nt << ", scale=" << scale << ")\n";
	std::cout << "dim,n,min_pts,optics_ms,gauss_ms,gauss_rand,struct_ms,struct_rand\n";
	run<3>( 8, 400 * scale, 10, nt );
	run<16>( 8, 400 * scale, 10, nt );
	run<64>( 8, 400 * scale, 10, nt );
	run<128>( 8, 400 * scale, 10, nt );
	return 0;
}
