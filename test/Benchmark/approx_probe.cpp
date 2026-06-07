// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Why is the approximate backend rarely faster than exact? This probe answers it with
// data: for each CSV cloud it reports the average neighborhood size, then for the exact
// backend and for the approximate backend at several eps levels it reports the full
// ordering time AND the neighbor recall (fraction of the exact eps-neighbors the
// approximate search still returns). When recall stays ~1.0 the approximate search did
// the same work, so it cannot be faster -- which is exactly what happens in low
// dimensions (the KD-tree boundary band the eps-pruning targets is negligible there).
//
// Build in a Release config. Usage: optics_approx_probe a.csv [b.csv ...] [min_pts]

#include <optics/optics.hpp>
#include <optics/Stopwatch.hpp>

#include "bench_config.hpp"
#include "csv_points.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace sw = stopwatch;

namespace {

template <std::size_t Dim>
double ordering_ms( const std::vector<std::array<double, Dim>>& pts, std::size_t min_pts,
					unsigned nt, unsigned permille ) {
	sw::Stopwatch w;
	if ( permille == 0 ) {
		(void)optics::compute_reachability_dists<double, Dim, optics::NanoflannBackend<double, Dim>>(
			pts, min_pts, -1.0, optics::NeighborMode::Precompute, nt );
	} else if ( permille == 100 ) {
		(void)optics::compute_reachability_dists<double, Dim, optics::NanoflannBackend<double, Dim, 100>>(
			pts, min_pts, -1.0, optics::NeighborMode::Precompute, nt );
	} else if ( permille == 500 ) {
		(void)optics::compute_reachability_dists<double, Dim, optics::NanoflannBackend<double, Dim, 500>>(
			pts, min_pts, -1.0, optics::NeighborMode::Precompute, nt );
	} else {
		(void)optics::compute_reachability_dists<double, Dim, optics::NanoflannBackend<double, Dim, 1000>>(
			pts, min_pts, -1.0, optics::NeighborMode::Precompute, nt );
	}
	return static_cast<double>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
}

// Mean recall + mean neighbor count of an approximate backend vs the exact one, over a
// sample of query points (every `stride`-th point).
template <std::size_t Dim, unsigned Permille>
void approx_quality( const std::vector<std::array<double, Dim>>& pts, double eps, std::size_t stride,
					 double& recall, double& avg_nbrs ) {
	const optics::NanoflannBackend<double, Dim> exact( pts );
	const optics::NanoflannBackend<double, Dim, Permille> approx( pts );
	const double r = static_cast<double>( eps );
	double recall_sum = 0.0, nbr_sum = 0.0;
	std::size_t q = 0;
	std::vector<std::size_t> a, b;
	for ( std::size_t i = 0; i < pts.size(); i += stride, ++q ) {
		a.clear();
		b.clear();
		exact.radius_search( pts[i], r, a );
		approx.radius_search( pts[i], r, b );
		const std::unordered_set<std::size_t> bs( b.begin(), b.end() );
		std::size_t hit = 0;
		for ( const std::size_t idx : a ) { if ( bs.count( idx ) ) { ++hit; } }
		recall_sum += a.empty() ? 1.0 : static_cast<double>( hit ) / static_cast<double>( a.size() );
		nbr_sum += static_cast<double>( b.size() );
	}
	recall = q ? recall_sum / static_cast<double>( q ) : 1.0;
	avg_nbrs = q ? nbr_sum / static_cast<double>( q ) : 0.0;
}

template <std::size_t Dim>
double exact_avg_nbrs( const std::vector<std::array<double, Dim>>& pts, double eps, std::size_t stride ) {
	const optics::NanoflannBackend<double, Dim> exact( pts );
	double sum = 0.0;
	std::size_t q = 0;
	std::vector<std::size_t> a;
	for ( std::size_t i = 0; i < pts.size(); i += stride, ++q ) {
		a.clear();
		exact.radius_search( pts[i], static_cast<double>( eps ), a );
		sum += static_cast<double>( a.size() );
	}
	return q ? sum / static_cast<double>( q ) : 0.0;
}

template <std::size_t Dim>
void probe( const std::string& name, const std::vector<double>& flat, std::size_t n,
			std::size_t min_pts, unsigned nt ) {
	const auto pts = bench::pack<Dim>( flat, n );
	const double eps = optics::epsilon_estimation( pts, min_pts );
	const std::size_t stride = std::max<std::size_t>( 1, n / 1500 );  // ~1500 query samples
	const double avg = exact_avg_nbrs<Dim>( pts, eps, stride );

	std::printf( "\n%s  n=%zu dim=%zu  eps=%.3g  avg_nbrs=%.0f\n",
				 name.c_str(), n, Dim, eps, avg );
	std::printf( "  %-16s %10s %8s %10s\n", "backend", "order_ms", "recall", "ap_nbrs" );
	std::printf( "  %-16s %10.0f %8.3f %10.0f\n", "exact", ordering_ms<Dim>( pts, min_pts, nt, 0 ), 1.0, avg );

	double rec = 0.0, an = 0.0;
	approx_quality<Dim, 100>( pts, eps, stride, rec, an );
	std::printf( "  %-16s %10.0f %8.3f %10.0f\n", "approx eps=0.1", ordering_ms<Dim>( pts, min_pts, nt, 100 ), rec, an );
	approx_quality<Dim, 500>( pts, eps, stride, rec, an );
	std::printf( "  %-16s %10.0f %8.3f %10.0f\n", "approx eps=0.5", ordering_ms<Dim>( pts, min_pts, nt, 500 ), rec, an );
	approx_quality<Dim, 1000>( pts, eps, stride, rec, an );
	std::printf( "  %-16s %10.0f %8.3f %10.0f\n", "approx eps=1.0", ordering_ms<Dim>( pts, min_pts, nt, 1000 ), rec, an );
}

}  // namespace

int main( int argc, char** argv ) {
	std::vector<std::string> paths;
	std::size_t min_pts = 10;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		if ( !a.empty() && a.find_first_not_of( "0123456789" ) == std::string::npos ) {
			min_pts = static_cast<std::size_t>( std::stoul( a ) );
		} else {
			paths.push_back( a );
		}
	}
	if ( paths.empty() ) {
		std::cerr << "usage: optics_approx_probe a.csv [b.csv ...] [min_pts]\n";
		return 2;
	}
	const unsigned nt = bench::threads();
	std::printf( "approx probe (threads=%u, min_pts=%zu) -- recall is the fraction of exact\n"
				 "eps-neighbors the approximate search still returns; ~1.0 => same work as exact.\n",
				 nt, min_pts );
	for ( const auto& path : paths ) {
		std::vector<double> flat;
		std::size_t n = 0, dim = 0;
		if ( !bench::read_csv( path, flat, n, dim ) ) { std::cerr << "skip: " << path << "\n"; continue; }
		switch ( dim ) {
			case 2:  probe<2>( path, flat, n, min_pts, nt ); break;
			case 3:  probe<3>( path, flat, n, min_pts, nt ); break;
			case 16: probe<16>( path, flat, n, min_pts, nt ); break;
			default: std::cerr << "skip (unsupported dim " << dim << "): " << path << "\n"; break;
		}
	}
	return 0;
}
