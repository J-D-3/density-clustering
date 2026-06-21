// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Emits this library's predicted HDBSCAN* labels (exact hdbscan and approximate shdbscan) for one
// CSV cloud, so a Python harness can score them against scikit-learn's HDBSCAN and ground truth
// (ARI / NMI / Rand). Thin on purpose: the datasets, the scikit-learn reference, and the metrics
// live in tools/hdbscan_benchmark.py.
//
// Both use the same (min_cluster_size, min_samples); min_samples is SELF-INCLUSIVE, matching
// sklearn.cluster.HDBSCAN, so the Python harness can pass identical parameters. shdbscan is a
// COSINE method (points L2-normalized internally), so on a raw-Euclidean layout it is expected to
// score lower than exact hdbscan -- the harness reports that honestly.
//
// stdout (labels):  point_index,hdbscan,shdbscan   (one row per point; -1 = noise)
// stderr (timing):  TIMING n=<n> dim=<Dim> hdbscan_ms=<..> shdbscan_ms=<..>
//
// Usage: hdbscan_compare coords.csv [min_cluster_size=15] [min_samples=0]
//   min_samples 0 => use min_cluster_size (sklearn's default). Build in a Release config; not a ctest.

#include <optics/hdbscan.hpp>
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
void run( const std::vector<double>& flat, std::size_t n, std::size_t mcs, std::size_t ms ) {
	const auto pts = bench::pack<Dim>( flat, n );

	sw::Stopwatch w1;
	const auto h = optics::hdbscan<double, Dim>( pts, mcs, ms );
	const long long h_ms = static_cast<long long>( bench::ceil_ms_from_us( w1.elapsed<sw::mus>() ) );

	sw::Stopwatch w2;
	const auto s = optics::shdbscan<double, Dim>( pts, mcs, ms );
	const long long s_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );

	std::cout << "point_index,hdbscan,shdbscan\n";
	for ( std::size_t i = 0; i < n; ++i ) {
		std::cout << i << "," << h.labels[i] << "," << s.labels[i] << "\n";
	}
	std::cout.flush();
	std::cerr << "TIMING n=" << n << " dim=" << Dim
			  << " hdbscan_ms=" << h_ms << " shdbscan_ms=" << s_ms << "\n";
}

}  // namespace

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::cerr << "usage: hdbscan_compare coords.csv [min_cluster_size=15] [min_samples=0]\n";
		return 2;
	}
	const std::string path = argv[1];
	const std::size_t mcs = ( argc > 2 ) ? static_cast<std::size_t>( std::stoul( argv[2] ) ) : 15;
	const std::size_t ms = ( argc > 3 ) ? static_cast<std::size_t>( std::stoul( argv[3] ) ) : 0;

	std::vector<double> flat;
	std::size_t n = 0, dim = 0;
	if ( !bench::read_csv( path, flat, n, dim ) ) { std::cerr << "cannot read " << path << "\n"; return 2; }
	switch ( dim ) {
		case 2:  run<2>( flat, n, mcs, ms ); break;
		case 3:  run<3>( flat, n, mcs, ms ); break;
		case 4:  run<4>( flat, n, mcs, ms ); break;
		case 8:  run<8>( flat, n, mcs, ms ); break;
		case 16: run<16>( flat, n, mcs, ms ); break;
		default: std::cerr << "unsupported dim " << dim << " (supported: 2,3,4,8,16)\n"; return 2;
	}
	return 0;
}
