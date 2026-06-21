// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Unique-point / weighted OPTICS speedup (issue #46), on CSV point clouds. Images repeat the
// same colors thousands of times in flat regions; deduplicating them to unique points + integer
// weights and running weight-aware OPTICS on the small unique cloud gives the SAME clustering far
// faster (a flat region of N identical pixels collapses to one point, so its O(neighborhood)
// ordering cost vanishes). This harness measures the win: full-cloud ordering vs deduplicate + the
// weighted ordering on the unique cloud, at the SAME epsilon (so the comparison is fair).
//
// Emits CSV to stdout:
//   dataset,n,unique,collapse,eps,full_ms,dedup_ms,weighted_ms,speedup
// where dedup_ms is the deduplication pass alone and weighted_ms the ordering on the unique cloud;
// the fair end-to-end comparison is full_ms vs (dedup_ms + weighted_ms), reported as speedup.
// Build in a Release config; not a ctest (timings vary by machine).
//
// Usage: optics_dedup_probe a.csv [b.csv ...] [min_pts]
//   (a bare integer arg sets min_pts; default 25, matching the color example)

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

template <std::size_t Dim>
void run( const std::string& name, const std::vector<double>& flat, std::size_t n, std::size_t min_pts ) {
	const auto pts = bench::pack<Dim>( flat, n );
	// Fix epsilon for BOTH paths so we measure the cloud-size effect, not an auto-eps difference
	// (the weighted path's auto-eps is the uniform estimate, the full path's is the knee).
	const double eps = optics::epsilon_estimation( pts, min_pts );

	sw::Stopwatch w_full;
	(void)optics::compute_reachability_dists<double, Dim>( pts, min_pts, eps, optics::NeighborMode::OnDemand, 1 );
	const long long full_ms = static_cast<long long>( bench::ceil_ms_from_us( w_full.elapsed<sw::mus>() ) );

	sw::Stopwatch w_dedup;
	const auto d = optics::deduplicate( pts );
	const long long dedup_ms = static_cast<long long>( bench::ceil_ms_from_us( w_dedup.elapsed<sw::mus>() ) );

	sw::Stopwatch w_weighted;
	(void)optics::compute_reachability_dists<double, Dim>(
		d.unique_points, min_pts, eps, optics::NeighborMode::OnDemand, 1, optics::CoreDistMode::Scan, 0, d.weights );
	const long long weighted_ms = static_cast<long long>( bench::ceil_ms_from_us( w_weighted.elapsed<sw::mus>() ) );

	const double collapse = static_cast<double>( n ) / static_cast<double>( std::max<std::size_t>( d.unique_points.size(), 1 ) );
	const double speedup = static_cast<double>( full_ms ) / static_cast<double>( std::max<long long>( dedup_ms + weighted_ms, 1 ) );
	std::cout << name << "," << n << "," << d.unique_points.size() << "," << collapse << "," << eps << ","
			  << full_ms << "," << dedup_ms << "," << weighted_ms << "," << speedup << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	std::vector<std::string> paths;
	std::size_t min_pts = 25;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		if ( !a.empty() && a.find_first_not_of( "0123456789" ) == std::string::npos ) {
			min_pts = static_cast<std::size_t>( std::stoul( a ) );
		} else { paths.push_back( a ); }
	}
	if ( paths.empty() ) {
		std::cerr << "usage: optics_dedup_probe a.csv [b.csv ...] [min_pts]\n";
		return 2;
	}
	std::cerr << "dedup probe (min_pts=" << min_pts << "; speedup = full_ms / (dedup_ms + weighted_ms))\n";
	std::cout << "dataset,n,unique,collapse,eps,full_ms,dedup_ms,weighted_ms,speedup\n";
	for ( const auto& path : paths ) {
		std::vector<double> flat;
		std::size_t n = 0, dim = 0;
		if ( !bench::read_csv( path, flat, n, dim ) ) { std::cerr << "skip: " << path << "\n"; continue; }
		switch ( dim ) {
			case 2:  run<2>( path, flat, n, min_pts ); break;
			case 3:  run<3>( path, flat, n, min_pts ); break;
			case 4:  run<4>( path, flat, n, min_pts ); break;
			default: std::cerr << "skip (unsupported dim " << dim << "): " << path << "\n"; break;
		}
	}
	return 0;
}
