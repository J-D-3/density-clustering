// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
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

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

	// Read the RGB cloud.
	std::vector<std::array<float, 3>> points;
	std::ifstream in( in_path );
	if ( !in ) { std::cerr << "cannot open " << in_path << "\n"; return 1; }
	std::string line;
	while ( std::getline( in, line ) ) {
		if ( line.empty() ) { continue; }
		std::array<float, 3> p{};
		std::size_t k = 0;
		std::stringstream ss( line );
		std::string tok;
		bool ok = true;
		while ( k < 3 && std::getline( ss, tok, ',' ) ) {
			try { p[k] = std::stof( tok ); } catch ( ... ) { ok = false; break; }
			++k;
		}
		if ( ok && k == 3 ) { points.push_back( p ); }  // non-numeric (header) rows are skipped
	}
	std::cout << "loaded " << points.size() << " RGB points from " << in_path << "\n";
	if ( points.empty() ) { return 1; }

	// Cluster the color space.
	const auto reach = optics::compute_reachability_dists( points, min_pts, eps );
	const auto clusters = optics::get_cluster_indices( reach, threshold );
	const std::size_t min_size = static_cast<std::size_t>( min_cluster_frac * static_cast<double>( points.size() ) );
	const auto labels = optics::io::cluster_labels( points.size(), clusters, min_size );

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
