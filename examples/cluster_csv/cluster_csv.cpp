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

// Split a CSV line into trimmed tokens.
std::vector<std::string> split( const std::string& line ) {
	std::vector<std::string> out;
	std::stringstream ss( line );
	std::string tok;
	while ( std::getline( ss, tok, ',' ) ) { out.push_back( tok ); }
	return out;
}

bool is_number( const std::string& s ) {
	if ( s.empty() ) { return false; }
	try { std::size_t n = 0; (void)std::stod( s, &n ); return n == s.size(); }
	catch ( ... ) { return false; }
}

// Run OPTICS for a compile-time dimension over a flat row-major coordinate buffer.
template <std::size_t Dim>
int run( const std::vector<double>& flat, std::size_t n, const std::string& out_prefix,
		 std::size_t min_pts, double eps, double threshold, double min_cluster_frac ) {
	std::vector<std::array<double, Dim>> points( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		for ( std::size_t d = 0; d < Dim; ++d ) { points[i][d] = flat[i * Dim + d]; }
	}

	const auto t0 = clk::now();
	const auto reach = optics::compute_reachability_dists( points, min_pts, eps );
	const auto t1 = clk::now();
	const auto clusters = optics::get_cluster_indices( reach, threshold );
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
		std::cerr << "usage: cluster_csv in.csv [out_prefix] [min_pts] [eps] [threshold] [min_cluster_frac]\n";
		return 2;
	}
	const std::string in_path = argv[1];
	const std::string out_prefix = ( argc > 2 ) ? argv[2] : "clustered";
	const std::size_t min_pts = ( argc > 3 ) ? static_cast<std::size_t>( std::atoi( argv[3] ) ) : 10;
	const double eps = ( argc > 4 ) ? std::atof( argv[4] ) : -1.0;
	const double threshold = ( argc > 5 ) ? std::atof( argv[5] ) : 2.0;
	const double min_cluster_frac = ( argc > 6 ) ? std::atof( argv[6] ) : 0.01;

	std::ifstream in( in_path );
	if ( !in ) { std::cerr << "cannot open " << in_path << "\n"; return 1; }

	// Determine the dimension. If the first line is non-numeric it is a header;
	// count its x-prefixed columns. Otherwise infer from the field count.
	std::string line;
	std::size_t dim = 0;
	std::vector<double> flat;
	std::size_t n = 0;
	if ( std::getline( in, line ) ) {
		const auto tok = split( line );
		bool header = false;
		for ( const auto& t : tok ) { if ( !is_number( t ) ) { header = true; break; } }
		if ( header ) {
			for ( const auto& t : tok ) { if ( !t.empty() && t[0] == 'x' ) { ++dim; } }
			if ( dim == 0 ) { dim = tok.size(); }  // header without x* names
		} else {
			dim = tok.size();
			// the first line is data: parse it now
			for ( std::size_t d = 0; d < dim; ++d ) { flat.push_back( std::stod( tok[d] ) ); }
			++n;
		}
	}
	if ( dim == 0 ) { std::cerr << "could not determine dimension from " << in_path << "\n"; return 1; }

	while ( std::getline( in, line ) ) {
		if ( line.empty() ) { continue; }
		const auto tok = split( line );
		if ( tok.size() < dim ) { continue; }
		bool ok = true;
		const std::size_t base = flat.size();
		for ( std::size_t d = 0; d < dim; ++d ) {
			if ( !is_number( tok[d] ) ) { ok = false; break; }
			flat.push_back( std::stod( tok[d] ) );
		}
		if ( ok ) { ++n; } else { flat.resize( base ); }
	}

	std::cout << "loaded " << n << " points (" << dim << "-D) from " << in_path << "\n";
	if ( n == 0 ) { return 1; }

	switch ( dim ) {
		case 2:  return run<2>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac );
		case 3:  return run<3>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac );
		case 4:  return run<4>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac );
		case 16: return run<16>( flat, n, out_prefix, min_pts, eps, threshold, min_cluster_frac );
		default:
			std::cerr << "unsupported dimension " << dim << " (supported: 2, 3, 4, 16)\n";
			return 1;
	}
}
