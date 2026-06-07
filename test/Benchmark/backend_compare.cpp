// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Times the OPTICS ordering across the internal neighbor-search backends on one or
// more CSV point clouds (the same files scikit-learn is timed on by
// tools/timing_compare.py). Emits machine-readable CSV to stdout:
//
//   dataset,n,dim,backend,ms
//
// Build in a Release config. Usage: optics_backend_compare a.csv b.csv [min_pts]
// (a trailing all-digits argument is taken as min_pts; default 10). Threads default
// to 4 (override with OPTICS_BENCH_THREADS). Boost is included only when built with
// -DOPTICS_ENABLE_BOOST_RTREE=ON.

#include <optics/optics.hpp>
#include <optics/Stopwatch.hpp>

#include "bench_config.hpp"
#include "csv_points.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace sw = stopwatch;

namespace {

template <class Backend, std::size_t Dim>
void time_backend( const std::string& dataset, const std::string& backend,
				   const std::vector<std::array<double, Dim>>& pts, std::size_t min_pts, unsigned nt ) {
	sw::Stopwatch w;
	const auto reach = optics::compute_reachability_dists<double, Dim, Backend>(
		pts, min_pts, -1.0, optics::NeighborMode::Precompute, nt );
	const auto ms = bench::ceil_ms_from_us( w.elapsed<sw::mus>() );
	(void)reach;
	std::cout << dataset << "," << pts.size() << "," << Dim << "," << backend << "," << ms << "\n";
}

template <std::size_t Dim>
void run_all_backends( const std::string& dataset, const std::vector<double>& flat,
					   std::size_t n, std::size_t min_pts, unsigned nt ) {
	const auto pts = bench::pack<Dim>( flat, n );
	time_backend<optics::NanoflannBackend<double, Dim>, Dim>( dataset, "nanoflann", pts, min_pts, nt );
	time_backend<optics::ApproxNanoflannBackend<double, Dim>, Dim>( dataset, "nf-approx", pts, min_pts, nt );
#ifdef OPTICS_ENABLE_BOOST_RTREE
	time_backend<optics::BoostRTreeBackend<double, Dim>, Dim>( dataset, "boost-rtree", pts, min_pts, nt );
#endif
}

}  // namespace

int main( int argc, char** argv ) {
	std::vector<std::string> paths;
	std::size_t min_pts = 10;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		const bool all_digits = !a.empty() && a.find_first_not_of( "0123456789" ) == std::string::npos;
		if ( all_digits ) { min_pts = static_cast<std::size_t>( std::stoul( a ) ); }
		else { paths.push_back( a ); }
	}
	if ( paths.empty() ) {
		std::cerr << "usage: optics_backend_compare a.csv [b.csv ...] [min_pts]\n";
		return 2;
	}

	const unsigned nt = bench::threads();
	std::cout << "dataset,n,dim,backend,ms\n";
	for ( const auto& path : paths ) {
		std::vector<double> flat;
		std::size_t n = 0, dim = 0;
		if ( !bench::read_csv( path, flat, n, dim ) ) { std::cerr << "skip (unreadable): " << path << "\n"; continue; }
		switch ( dim ) {
			case 2:  run_all_backends<2>( path, flat, n, min_pts, nt ); break;
			case 3:  run_all_backends<3>( path, flat, n, min_pts, nt ); break;
			case 4:  run_all_backends<4>( path, flat, n, min_pts, nt ); break;
			case 16: run_all_backends<16>( path, flat, n, min_pts, nt ); break;
			default: std::cerr << "skip (unsupported dim " << dim << "): " << path << "\n"; break;
		}
	}
	return 0;
}
