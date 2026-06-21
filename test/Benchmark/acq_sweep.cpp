// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// OPTICS acquisition-knob crossover sweep (issue #72): grounds the NeighborMode::Auto and
// CoreDistMode::Auto thresholds. For one synthetic cloud it sweeps epsilon (hence the average
// neighborhood size) and, at each density, times the metric-preserving acquisition options:
//   - CoreDistMode: OnDemand+Scan vs OnDemand+Knn  -> where does Knn (byte-identical) overtake Scan?
//   - NeighborMode: OnDemand+Scan vs Precompute+Scan -> confirms Precompute wins wherever its cache
//                   fits (the matrix D3 finding; the Auto rule is a pure cache-fit check).
// All four orderings are identical (Knn and the modes only change *how* neighbors / core distances
// are obtained), so this is purely a timing study.
//
// Emits CSV to stdout:
//   n,dim,eps,avg_nbrs,od_scan_ms,od_knn_ms,pc_scan_ms,coredist_winner,mode_winner
//
// Usage: optics_acq_sweep [scale]   (scale multiplies the point count; default 1).
// Build in a Release config; not a ctest (timings vary by machine).

#include <optics/optics.hpp>
#include <optics/Stopwatch.hpp>
#include <optics/testdata.hpp>

#include "bench_config.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

namespace sw = stopwatch;

namespace {

template <std::size_t Dim>
double avg_neighbors( const std::vector<std::array<double, Dim>>& pts, double eps ) {
	const optics::NanoflannBackend<double, Dim> backend( pts );
	const std::size_t sample = std::min<std::size_t>( pts.size(), 256 );
	const std::size_t stride = std::max<std::size_t>( 1, pts.size() / sample );
	double sum = 0.0;
	std::size_t q = 0;
	std::vector<std::size_t> buf;
	for ( std::size_t i = 0; i < pts.size(); i += stride, ++q ) {
		buf.clear();
		backend.radius_search( pts[i], eps, buf );
		sum += static_cast<double>( buf.size() );
	}
	return q ? sum / static_cast<double>( q ) : 0.0;
}

template <std::size_t Dim>
long long time_ms( const std::vector<std::array<double, Dim>>& pts, std::size_t min_pts, double eps,
				   optics::NeighborMode mode, optics::CoreDistMode cd, unsigned nt ) {
	sw::Stopwatch w;
	(void)optics::compute_reachability_dists<double, Dim>( pts, min_pts, eps, mode, nt, cd );
	return static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
}

template <std::size_t Dim>
void run( std::size_t n_blobs, std::size_t per, std::size_t min_pts, unsigned nt ) {
	// Tight blobs (stddev 1) widely spaced (spread 30): sweeping eps from ~1 to ~12 grows each point's
	// neighborhood from a handful to most of its blob, spanning sparse -> very dense.
	const auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per, 30.0, 1.0, 321u );
	const std::size_t n = pts.size();
	for ( const double eps : { 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0 } ) {
		const double an = avg_neighbors<Dim>( pts, eps );
		const long long od_scan = time_ms<Dim>( pts, min_pts, eps, optics::NeighborMode::OnDemand, optics::CoreDistMode::Scan, nt );
		const long long od_knn = time_ms<Dim>( pts, min_pts, eps, optics::NeighborMode::OnDemand, optics::CoreDistMode::Knn, nt );
		const long long pc_scan = time_ms<Dim>( pts, min_pts, eps, optics::NeighborMode::Precompute, optics::CoreDistMode::Scan, nt );
		const char* cd_win = ( od_knn < od_scan ) ? "knn" : "scan";
		const char* mode_win = ( pc_scan < od_scan ) ? "precompute" : "ondemand";
		std::cout << n << "," << Dim << "," << eps << "," << an << "," << od_scan << "," << od_knn << ","
				  << pc_scan << "," << cd_win << "," << mode_win << "\n";
		std::cout.flush();
	}
}

}  // namespace

int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }
	const unsigned nt = bench::threads();
	std::cerr << "OPTICS acquisition sweep (threads=" << nt << ", scale=" << scale << "). Maps avg_nbrs -> "
			  << "fastest CoreDistMode (Scan/Knn, byte-identical) and NeighborMode (Precompute where its "
			  << "cache fits, else OnDemand).\n";
	std::cout << "n,dim,eps,avg_nbrs,od_scan_ms,od_knn_ms,pc_scan_ms,coredist_winner,mode_winner\n";
	run<3>( 8, 2500 * scale, 10, nt );
	run<8>( 7, 2800 * scale, 10, nt );
	run<16>( 6, 1500 * scale, 10, nt );
	return 0;
}
