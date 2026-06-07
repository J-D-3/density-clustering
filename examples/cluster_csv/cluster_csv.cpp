// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Cluster *your own* point cloud with OPTICS. Reads a coordinates CSV (a header
// row of x0,x1,...; one numeric row per point; any extra trailing columns are
// ignored), runs OPTICS, and writes two CSVs for tools/visualize.py:
//   <out>_points.csv  -- x0,...,x{Dim-1},cluster_id   (the labeled cloud)
//   <out>_reach.csv   -- order_index,point_index,reachability  (the plot)
//
// The dimension is detected from the header and dispatched to a fixed set of
// supported dims (2, 3, 4, 16). Generate ready-made inputs with tools/datasets.py.
//
// Usage: cluster_csv in.csv [out_prefix] [min_pts] [epsilon] [threshold] [min_cluster_frac]
//   epsilon <= 0 auto-estimates; threshold cuts the reachability plot into clusters.

#include <optics/optics.hpp>
#include <optics/io.hpp>

#include "../shared/csv_io.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
// Elapsed time rounded UP to the next whole millisecond.
long long ms( clk::time_point a, clk::time_point b ) {
	const auto us = std::chrono::duration_cast<std::chrono::microseconds>( b - a ).count();
	return ( us + 999 ) / 1000;
}

// Run OPTICS for a compile-time dimension over a flat row-major coordinate buffer.
// xi_chi > 0 selects the hierarchical Xi (steep-area) extraction instead of the flat
// threshold cut -- it recovers clusters at differing densities that one threshold can't.
template <std::size_t Dim>
int run( const std::vector<double>& flat, std::size_t n, const std::string& out_prefix,
		 std::size_t min_pts, double eps, double threshold, double min_cluster_frac,
		 unsigned n_threads, double xi_chi ) {
	std::vector<std::array<double, Dim>> points( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		for ( std::size_t d = 0; d < Dim; ++d ) { points[i][d] = flat[i * Dim + d]; }
	}

	// OnDemand by default (lean memory, faster on dense clouds); n_threads > 1 opts
	// into the parallel Precompute cache (faster on sparse/low-density clouds).
	const auto mode = ( n_threads > 1 ) ? optics::NeighborMode::Precompute : optics::NeighborMode::OnDemand;
	const auto t0 = clk::now();
	const auto reach = optics::compute_reachability_dists( points, min_pts, eps, mode, n_threads );
	const auto t1 = clk::now();
	const auto clusters = ( xi_chi > 0.0 )
		? optics::get_cluster_indices( reach, optics::get_chi_clusters_flat( reach, xi_chi, min_pts ) )
		: optics::get_cluster_indices( reach, threshold );
	const std::size_t min_size = static_cast<std::size_t>( min_cluster_frac * static_cast<double>( n ) );
	const auto labels = optics::io::cluster_labels( n, clusters, min_size );
	const auto t2 = clk::now();

	const std::string pts_path = out_prefix + "_points.csv";
	const std::string reach_path = out_prefix + "_reach.csv";
	optics::io::export_points_csv( pts_path, points, labels );
	optics::io::export_reachability_csv( reach_path, reach );

	long long max_label = -1;
	for ( const long long l : labels ) { max_label = std::max( max_label, l ); }
	std::size_t noise = 0;
	for ( const long long l : labels ) { if ( l < 0 ) { ++noise; } }

	std::cout << "OPTICS ordering " << ms( t0, t1 ) << " ms | extract " << ms( t1, t2 ) << " ms\n";
	std::cout << "clusters (>= " << min_size << " pts): " << ( max_label + 1 )
			  << " | noise: " << noise << " / " << n << "\n";
	std::cout << "wrote " << pts_path << " and " << reach_path << "\n";
	std::cout << "render: python tools/visualize.py --points " << pts_path << " --reach " << reach_path << "\n";
	return 0;
}

}  // namespace

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::cerr << "usage: cluster_csv in.csv [out_prefix] [min_pts] [eps] [threshold] [min_cluster_frac] [xi_chi] [n_threads]\n"
				  << "  xi_chi > 0 uses the hierarchical Xi extraction (varying densities).\n"
				  << "  n_threads: 1 (default) = lean OnDemand; >1 = parallel Precompute with that many threads.\n";
		return 2;
	}
	const std::string in_path = argv[1];
	const std::string out_prefix = ( argc > 2 ) ? argv[2] : "clustered";
	const std::size_t min_pts = ( argc > 3 ) ? static_cast<std::size_t>( std::atoi( argv[3] ) ) : 10;
	const double eps = ( argc > 4 ) ? std::atof( argv[4] ) : -1.0;
	const double threshold = ( argc > 5 ) ? std::atof( argv[5] ) : 2.0;
	const double min_cluster_frac = ( argc > 6 ) ? std::atof( argv[6] ) : 0.01;
	const double xi_chi = ( argc > 7 ) ? std::atof( argv[7] ) : 0.0;
	const unsigned n_threads = ( argc > 8 ) ? static_cast<unsigned>( std::atoi( argv[8] ) ) : 1u;

	// Read the cloud (dimension detected from the header / field count).
	std::vector<double> flat;
	std::size_t n = 0, dim = 0;
	if ( !example_io::read_csv( in_path, flat, n, dim ) ) {
		std::cerr << "could not read points from " << in_path << "\n";
		return 1;
	}
	std::cout << "loaded " << n << " points (" << dim << "-D) from " << in_path << "\n";

	switch ( dim ) {
		case 2:  return run<2>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac, n_threads, xi_chi );
		case 3:  return run<3>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac, n_threads, xi_chi );
		case 4:  return run<4>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac, n_threads, xi_chi );
		case 16: return run<16>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac, n_threads, xi_chi );
		default:
			std::cerr << "unsupported dimension " << dim << " (supported: 2, 3, 4, 16)\n";
			return 1;
	}
}
