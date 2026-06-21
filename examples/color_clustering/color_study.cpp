// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// color_study -- a machine-readable, multi-algorithm color-clustering harness for the
// real-world image study (see study/color_clustering/). It is a generalization of
// optics_color.cpp: same CSV-in / labels-out shape, but it switches between the library's
// clustering algorithms and prints a single JSON line of metrics on stdout so a Python
// orchestrator can drive a whole matrix of (image x space x algorithm x params) runs.
//
// It is deliberately SPACE-AGNOSTIC: it clusters whatever 3 numeric columns it is given
// (RGB or CIELAB -- the orchestrator decides), in Euclidean distance. All human-readable,
// color-aware reporting (per-cluster mean color, recall of expected colors) is done in
// Python from the per-row labels this tool writes.
//
// Pipeline (uniform across algorithms, so phase timings are comparable):
//   read CSV -> [optional voxel quantize] -> deduplicate (unique points + weights, issue #46)
//            -> cluster the SMALL unique cloud weight-aware -> expand labels back per pixel
//            -> significance filter (drop clusters < min_cluster_frac of the pixels).
//
// Usage:
//   color_study <in.csv> --algo <name> [flags]
//   algos: optics-threshold | optics-xi | hdbscan | shdbscan | soptics
//   flags: --min-pts N --eps E --threshold T --xi X --min-cluster-size M --min-samples S
//          --voxel BIN --min-cluster-frac F --metric {l2,l1,cosine} --n-projections P
//          --seed S --labels-out PATH --reach-out PATH
// stdout: one JSON line of metrics. stderr: human-readable progress.

#include <optics/optics.hpp>
#include <optics/hdbscan.hpp>
#include <optics/io.hpp>
#include <optics/preprocess.hpp>

#include "../shared/csv_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double ms( clk::time_point a, clk::time_point b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); }

// Tiny flag parser: --key value (and --key for the few we don't use here). Unknown keys error.
struct Args {
	std::map<std::string, std::string> kv;
	std::string get( const std::string& k, const std::string& def ) const {
		auto it = kv.find( k );
		return it == kv.end() ? def : it->second;
	}
	double num( const std::string& k, double def ) const {
		auto it = kv.find( k );
		return it == kv.end() ? def : std::atof( it->second.c_str() );
	}
	long long inum( const std::string& k, long long def ) const {
		auto it = kv.find( k );
		return it == kv.end() ? def : std::atoll( it->second.c_str() );
	}
};

// Build per-cluster index groups (over the ORIGINAL cloud) from a per-point label vector
// where >=0 is a cluster id and <0 is noise. Used to funnel HDBSCAN's labels through the
// SAME significance filter (optics::io::cluster_labels) as the index-list algorithms, so
// every algorithm's reported cluster count means the same thing.
std::vector<std::vector<std::size_t>> groups_from_labels( const std::vector<int>& labels ) {
	int max_id = -1;
	for ( int l : labels ) { max_id = std::max( max_id, l ); }
	std::vector<std::vector<std::size_t>> groups( max_id + 1 );
	for ( std::size_t i = 0; i < labels.size(); ++i ) {
		if ( labels[i] >= 0 ) { groups[labels[i]].push_back( i ); }
	}
	return groups;
}
}  // namespace

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::cerr << "usage: color_study <in.csv> --algo <name> [flags]\n";
		return 2;
	}
	Args args;
	const std::string in_path = argv[1];
	for ( int i = 2; i < argc; ++i ) {
		std::string a = argv[i];
		if ( a.rfind( "--", 0 ) != 0 ) { std::cerr << "unexpected token: " << a << "\n"; return 2; }
		std::string key = a.substr( 2 );
		std::string val = ( i + 1 < argc && std::string( argv[i + 1] ).rfind( "--", 0 ) != 0 ) ? argv[++i] : "1";
		args.kv[key] = val;
	}

	const std::string algo = args.get( "algo", "" );
	const std::size_t min_pts = static_cast<std::size_t>( args.inum( "min-pts", 20 ) );
	const double eps = args.num( "eps", -1.0 );
	const double threshold = args.num( "threshold", -1.0 );
	const double xi = args.num( "xi", 0.05 );
	std::size_t min_cluster_size = static_cast<std::size_t>( args.inum( "min-cluster-size", 0 ) );
	const std::size_t min_samples = static_cast<std::size_t>( args.inum( "min-samples", 0 ) );
	const double voxel = args.num( "voxel", 0.0 );
	const double min_cluster_frac = args.num( "min-cluster-frac", 0.003 );
	const std::string metric_s = args.get( "metric", "l2" );
	const unsigned n_projections = static_cast<unsigned>( args.inum( "n-projections", 1024 ) );
	const unsigned seed = static_cast<unsigned>( args.inum( "seed", 42 ) );
	const std::string labels_out = args.get( "labels-out", "" );
	const std::string reach_out = args.get( "reach-out", "" );

	optics::Metric metric = optics::Metric::L2;
	if ( metric_s == "l1" ) { metric = optics::Metric::L1; }
	else if ( metric_s == "cosine" ) { metric = optics::Metric::Cosine; }

	// --- read RGB/Lab cloud ---
	const auto t_read0 = clk::now();
	std::vector<double> flat;
	std::size_t n_in = 0, dim = 0;
	if ( !example_io::read_csv( in_path, flat, n_in, dim ) || dim < 3 ) {
		std::cerr << "could not read a 3-column cloud from " << in_path << "\n";
		return 1;
	}
	auto points = example_io::pack<float, 3>( flat, n_in, dim );
	const auto t_read1 = clk::now();
	const std::size_t n = points.size();
	std::cerr << "loaded " << n << " points from " << in_path << " (algo=" << algo << ")\n";
	if ( points.empty() ) { return 1; }

	// --- optional voxel quantize (lossy) then deduplicate (lossless) ---
	if ( voxel > 0.0 ) { points = optics::quantize( points, voxel ); }
	const auto t_dedup0 = clk::now();
	const auto dd = optics::deduplicate( points );
	const auto t_dedup1 = clk::now();
	const double collapse = static_cast<double>( n ) / static_cast<double>( std::max<std::size_t>( dd.unique_points.size(), 1 ) );
	std::cerr << "dedup " << n << " -> " << dd.unique_points.size() << " unique (" << collapse << "x"
			  << ( voxel > 0.0 ? ", voxel bin=" + std::to_string( voxel ) : "" ) << ")\n";

	if ( min_cluster_size == 0 ) { min_cluster_size = std::max<std::size_t>( min_pts, 2 ); }
	const std::size_t min_size = std::max<std::size_t>( 1, static_cast<std::size_t>( min_cluster_frac * static_cast<double>( n ) ) );

	// --- cluster the small unique cloud, weight-aware; produce per-ORIGINAL labels ---
	double t_core = 0.0, t_extract = 0.0;
	std::vector<long long> labels;
	std::vector<optics::reachability_dist> reach;  // kept for optional reachability export (OPTICS family)

	try {
		if ( algo == "optics-threshold" || algo == "soptics" ) {
			const auto c0 = clk::now();
			if ( algo == "optics-threshold" ) {
				reach = optics::compute_reachability_dists( dd.unique_points, min_pts, eps,
						optics::NeighborMode::OnDemand, 0, optics::CoreDistMode::Scan, 0, dd.weights );
			} else {
				reach = optics::compute_soptics_reachability_dists( dd.unique_points, min_pts, eps,
						n_projections, 0, 0, seed, 0, metric, 0.0, dd.weights );
			}
			const auto c1 = clk::now();
			const double t = ( threshold < 0.0 ) ? optics::detail::default_threshold( reach ) : threshold;
			const auto uclusters = optics::get_cluster_indices( reach, t );
			const auto clusters = optics::expand_clusters_to_original( uclusters, dd.unique_of_original );
			const auto c2 = clk::now();
			labels = optics::io::cluster_labels( n, clusters, min_size );
			t_core = ms( c0, c1 );
			t_extract = ms( c1, c2 );
		} else if ( algo == "optics-xi" ) {
			const auto c0 = clk::now();
			reach = optics::compute_reachability_dists( dd.unique_points, min_pts, eps,
					optics::NeighborMode::OnDemand, 0, optics::CoreDistMode::Scan, 0, dd.weights );
			const auto c1 = clk::now();
			std::vector<std::size_t> w_ord( reach.size() );
			for ( std::size_t i = 0; i < reach.size(); ++i ) { w_ord[i] = dd.weights[reach[i].point_index]; }
			const auto flat_xi = optics::get_chi_clusters_flat( reach, xi, min_pts, 0.0, 0, w_ord );
			const auto uclusters = optics::get_cluster_indices( reach, flat_xi );
			const auto clusters = optics::expand_clusters_to_original( uclusters, dd.unique_of_original );
			const auto c2 = clk::now();
			labels = optics::io::cluster_labels( n, clusters, min_size );
			t_core = ms( c0, c1 );
			t_extract = ms( c1, c2 );
		} else if ( algo == "hdbscan" || algo == "shdbscan" ) {
			const auto c0 = clk::now();
			optics::HdbscanResult res;
			if ( algo == "hdbscan" ) {
				res = optics::hdbscan( dd.unique_points, min_cluster_size, min_samples,
						optics::ClusterSelectionMethod::EOM, false, 0, /*dedup=*/false, dd.weights );
			} else {
				res = optics::shdbscan( dd.unique_points, min_cluster_size, min_samples, eps,
						n_projections, 0, 0, seed, 0, optics::ClusterSelectionMethod::EOM, false,
						metric, 0.0, optics::SopticsProjection::Gaussian, /*dedup=*/false, dd.weights );
			}
			const auto c1 = clk::now();
			// expand unique-point labels back to per-original, then funnel through the same filter.
			std::vector<int> orig_labels( n, -1 );
			for ( std::size_t o = 0; o < n; ++o ) { orig_labels[o] = res.labels[dd.unique_of_original[o]]; }
			labels = optics::io::cluster_labels( n, groups_from_labels( orig_labels ), min_size );
			t_core = ms( c0, c1 );
			t_extract = 0.0;
		} else {
			std::cerr << "unknown --algo '" << algo << "'\n";
			return 2;
		}
	} catch ( const std::exception& e ) {
		std::cerr << "clustering failed: " << e.what() << "\n";
		return 1;
	}

	// --- summarize ---
	long long max_label = -1;
	std::size_t noise = 0;
	for ( long long l : labels ) { max_label = std::max( max_label, l ); if ( l < 0 ) { ++noise; } }
	const long long n_clusters = max_label + 1;

	if ( !labels_out.empty() ) { optics::io::export_points_csv( labels_out, points, labels ); }
	if ( !reach_out.empty() && !reach.empty() ) { optics::io::export_reachability_csv( reach_out, reach ); }

	// One JSON line on stdout (everything else went to stderr).
	std::cout << "{"
			  << "\"algo\":\"" << algo << "\","
			  << "\"n\":" << n << ","
			  << "\"unique\":" << dd.unique_points.size() << ","
			  << "\"collapse\":" << collapse << ","
			  << "\"voxel\":" << voxel << ","
			  << "\"min_pts\":" << min_pts << ","
			  << "\"min_cluster_size\":" << min_cluster_size << ","
			  << "\"min_size_px\":" << min_size << ","
			  << "\"n_clusters\":" << n_clusters << ","
			  << "\"noise\":" << noise << ","
			  << "\"t_read_ms\":" << ms( t_read0, t_read1 ) << ","
			  << "\"t_dedup_ms\":" << ms( t_dedup0, t_dedup1 ) << ","
			  << "\"t_core_ms\":" << t_core << ","
			  << "\"t_extract_ms\":" << t_extract << ","
			  << "\"t_total_ms\":" << ( ms( t_read0, t_read1 ) + ms( t_dedup0, t_dedup1 ) + t_core + t_extract )
			  << "}" << std::endl;
	return 0;
}
