// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Color-space clustering demo: read an RGB cloud (CSV: r,g,b per row, optional
// header), cluster it with OPTICS, write a labeled CSV (x0,x1,x2,cluster_id),
// and print a per-cluster summary (size + mean color).
//
// Usage: optics_color in.csv out.csv [min_pts] [epsilon] [threshold] [min_cluster_frac]
//   epsilon <= 0 auto-estimates; min_cluster_frac filters noise from the summary.

#include <optics/optics.hpp>
#include <optics/io.hpp>

#include "../shared/csv_io.hpp"

#include <algorithm>
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
double ms( clk::time_point a, clk::time_point b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); }
}  // namespace

int main( int argc, char** argv ) {
	if ( argc < 3 ) {
		std::cerr << "usage: optics_color in.csv out.csv [min_pts] [eps] [threshold] [min_cluster_frac]\n";
		return 2;
	}
	const std::string in_path = argv[1];
	const std::string out_path = argv[2];
	const std::size_t min_pts = ( argc > 3 ) ? static_cast<std::size_t>( std::atoi( argv[3] ) ) : 20;
	const double eps = ( argc > 4 ) ? std::atof( argv[4] ) : -1.0;
	const double threshold = ( argc > 5 ) ? std::atof( argv[5] ) : 25.0;
	const double min_cluster_frac = ( argc > 6 ) ? std::atof( argv[6] ) : 0.01;

	// Read the RGB cloud (CSV: r,g,b per row, optional header; extra columns ignored).
	const auto t_read0 = clk::now();
	std::vector<double> flat;
	std::size_t n = 0, dim = 0;
	if ( !example_io::read_csv( in_path, flat, n, dim ) || dim < 3 ) {
		std::cerr << "could not read an RGB cloud (>= 3 columns) from " << in_path << "\n";
		return 1;
	}
	const auto points = example_io::pack<float, 3>( flat, n, dim );
	const auto t_read1 = clk::now();
	std::cout << "loaded " << points.size() << " RGB points from " << in_path << "\n";
	if ( points.empty() ) { return 1; }

	// Cluster the color space. Images have large flat-color regions, so we first DEDUPLICATE
	// identical pixels into unique colors carrying a weight (count) and run weight-aware OPTICS on
	// the small unique cloud (issue #46): a flat region of N identical pixels collapses to one
	// point, so its O(neighborhood) ordering cost vanishes. The clustering is lossless -- expanding
	// back gives the same per-pixel partition as clustering the full cloud, just far faster.
	const auto t_dedup0 = clk::now();
	const auto dedup = optics::deduplicate( points );
	const auto t_dedup1 = clk::now();
	const double collapse = static_cast<double>( points.size() ) / static_cast<double>( std::max<std::size_t>( dedup.unique_points.size(), 1 ) );
	std::cout << "deduplicated " << points.size() << " px -> " << dedup.unique_points.size()
			  << " unique colors (" << collapse << "x collapse)\n";

	const auto t_optics0 = clk::now();
	const auto reach = optics::compute_reachability_dists(
		dedup.unique_points, min_pts, eps, optics::NeighborMode::OnDemand, 0, optics::CoreDistMode::Scan, 0, dedup.weights );
	const auto t_optics1 = clk::now();
	const auto unique_clusters = optics::get_cluster_indices( reach, threshold );
	const auto clusters = optics::expand_clusters_to_original( unique_clusters, dedup.unique_of_original );
	const std::size_t min_size = static_cast<std::size_t>( min_cluster_frac * static_cast<double>( points.size() ) );
	const auto labels = optics::io::cluster_labels( points.size(), clusters, min_size );
	const auto t_extract = clk::now();

	std::cout << "timing: csv-read " << ms( t_read0, t_read1 ) << " ms | dedup " << ms( t_dedup0, t_dedup1 )
			  << " ms | OPTICS ordering " << ms( t_optics0, t_optics1 ) << " ms | extract "
			  << ms( t_optics1, t_extract ) << " ms\n";

	optics::io::export_points_csv( out_path, points, labels );

	// Summary: count and mean color of each non-noise cluster.
	long long max_label = -1;
	for ( long long l : labels ) { max_label = std::max( max_label, l ); }
	std::cout << "significant clusters (>= " << min_size << " px): " << ( max_label + 1 ) << "\n";
	for ( long long c = 0; c <= max_label; ++c ) {
		double r = 0, g = 0, b = 0;
		std::size_t n = 0;
		for ( std::size_t i = 0; i < points.size(); ++i ) {
			if ( labels[i] == c ) { r += points[i][0]; g += points[i][1]; b += points[i][2]; ++n; }
		}
		if ( n ) {
			std::cout << "  cluster " << c << ": " << n << " px  mean RGB ("
					  << static_cast<int>( r / n ) << ", " << static_cast<int>( g / n ) << ", " << static_cast<int>( b / n ) << ")\n";
		}
	}
	std::size_t noise = 0;
	for ( long long l : labels ) { if ( l < 0 ) ++noise; }
	std::cout << "  noise/edge px: " << noise << "\n";
	return 0;
}
