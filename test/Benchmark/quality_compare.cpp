// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Emits this library's predicted cluster labels (OPTICS and sOPTICS) for one CSV cloud,
// so a Python harness can score them against ground truth (ARI / NMI / Rand). Thin on
// purpose: the dataset generation, scikit-learn reference methods, and metrics live in
// tools/quality_benchmark.py.
//
// Both algorithms use Xi (steep-area) extraction at the same chi, so the comparison is
// extractor-consistent. sOPTICS uses the cosine metric (points are L2-normalized
// internally), so its labels are meaningful on cosine-appropriate data (e.g. the
// high-dimensional normalized clouds the Python harness adds), not arbitrary Euclidean
// layouts -- the harness reports both honestly.
//
// stdout (labels):  point_index,optics,soptics   (one row per point; -1 = noise)
// stderr (timing):  TIMING n=<n> dim=<Dim> eps=<generating distance> optics_ms=<..> soptics_ms=<..>
//   (eps is emitted so an external engine can be run at the same generating distance.)
//
// Usage: optics_quality_compare coords.csv [min_pts=10] [chi=0.05] [min_cluster_size=0] [eps=uniform|knee|<num>]
//   min_cluster_size 0 => Xi uses min_pts (issue #57). eps "knee" uses the k-distance-knee
//   estimator (better on clustered data); a number sets it explicitly; default is the
//   uniform-density estimate. Build in a Release config; not a ctest.

#include <optics/optics.hpp>
#include <optics/io.hpp>
#include <optics/Stopwatch.hpp>

#include "bench_config.hpp"
#include "csv_points.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace sw = stopwatch;

namespace {

template <std::size_t Dim>
void run( const std::vector<double>& flat, std::size_t n, std::size_t min_pts, double chi,
		  std::size_t min_cluster_size, const std::string& eps_spec ) {
	const auto pts = bench::pack<Dim>( flat, n );

	// Generating distance, computed outside the timed region and emitted so an external
	// engine (dbscan-R) can run at the *same* eps. eps_spec: "knee" (k-distance knee,
	// better on clustered data -- issue #57), a positive number (explicit), else the
	// uniform-density estimate (the library default).
	const double eps =
		( eps_spec == "knee" ) ? optics::epsilon_estimation_knee<double, Dim>( pts, min_pts )
		: ( std::atof( eps_spec.c_str() ) > 0.0 ) ? std::atof( eps_spec.c_str() )
		: optics::epsilon_estimation( pts, min_pts );

	sw::Stopwatch w1;
	const auto o_reach = optics::compute_reachability_dists<double, Dim>( pts, min_pts, eps );
	const long long o_ms = static_cast<long long>( bench::ceil_ms_from_us( w1.elapsed<sw::mus>() ) );
	const auto o_labels = optics::io::cluster_labels(
		n, optics::get_cluster_indices( o_reach, optics::get_chi_clusters_flat( o_reach, chi, min_pts, 0.0, min_cluster_size ) ) );

	sw::Stopwatch w2;
	const auto s_reach = optics::compute_soptics_reachability_dists<double, Dim>( pts, min_pts );
	const long long s_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );
	const auto s_labels = optics::io::cluster_labels(
		n, optics::get_cluster_indices( s_reach, optics::get_chi_clusters_flat( s_reach, chi, min_pts, 0.0, min_cluster_size ) ) );

	std::cout << "point_index,optics,soptics\n";
	for ( std::size_t i = 0; i < n; ++i ) {
		std::cout << i << "," << o_labels[i] << "," << s_labels[i] << "\n";
	}
	std::cout.flush();
	std::cerr << "TIMING n=" << n << " dim=" << Dim << " eps=" << eps
			  << " optics_ms=" << o_ms << " soptics_ms=" << s_ms << "\n";
}

}  // namespace

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::cerr << "usage: optics_quality_compare coords.csv [min_pts=10] [chi=0.05] [min_cluster_size=0] [eps=uniform|knee|<num>]\n";
		return 2;
	}
	const std::string path = argv[1];
	const std::size_t min_pts = ( argc > 2 ) ? static_cast<std::size_t>( std::stoul( argv[2] ) ) : 10;
	const double chi = ( argc > 3 ) ? std::stod( argv[3] ) : 0.05;
	// 0 => Xi uses min_pts (issue #57: decoupling the Xi minimum-cluster-size from min_pts).
	const std::size_t mcs = ( argc > 4 ) ? static_cast<std::size_t>( std::stoul( argv[4] ) ) : 0;
	const std::string eps_spec = ( argc > 5 ) ? argv[5] : "uniform";

	std::vector<double> flat;
	std::size_t n = 0, dim = 0;
	if ( !bench::read_csv( path, flat, n, dim ) ) { std::cerr << "cannot read " << path << "\n"; return 2; }
	switch ( dim ) {
		case 2:  run<2>( flat, n, min_pts, chi, mcs, eps_spec ); break;
		case 3:  run<3>( flat, n, min_pts, chi, mcs, eps_spec ); break;
		case 4:  run<4>( flat, n, min_pts, chi, mcs, eps_spec ); break;
		case 8:  run<8>( flat, n, min_pts, chi, mcs, eps_spec ); break;
		case 16: run<16>( flat, n, min_pts, chi, mcs, eps_spec ); break;
		default: std::cerr << "unsupported dim " << dim << " (supported: 2,3,4,8,16)\n"; return 2;
	}
	return 0;
}
