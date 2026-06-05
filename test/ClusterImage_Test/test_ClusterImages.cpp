// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Visual / inspection test: cluster synthetic (and optionally hand-drawn) point
// clouds, assert structure, and export CSVs for tools/visualize.py to render.

#include <optics/optics.hpp>
#include <optics/io.hpp>
#include <optics/testdata.hpp>

#include "../support/ppm_fixture.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>


namespace {

// Number of clusters with at least min_size members (ignores singleton noise).
std::size_t count_large_clusters( const std::vector<std::vector<std::size_t>>& clusters, std::size_t min_size ) {
	std::size_t n = 0;
	for ( const auto& c : clusters ) {
		if ( c.size() >= min_size ) { ++n; }
	}
	return n;
}

// Cluster a cloud and export points+labels and the reachability plot as CSV.
template <typename T, std::size_t Dim>
std::vector<std::vector<std::size_t>> cluster_and_export(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts, double threshold,
	const std::string& tag ) {
	const auto reach_dists = optics::compute_reachability_dists( points, min_pts );
	const auto clusters = optics::get_cluster_indices( reach_dists, threshold );
	const auto labels = optics::io::cluster_labels( points.size(), clusters, /*min_cluster_size=*/min_pts );

	optics::io::export_points_csv( tag + "_points.csv", points, labels );
	optics::io::export_reachability_csv( tag + "_reachability.csv", reach_dists );
	std::cout << "  wrote " << tag << "_points.csv and " << tag << "_reachability.csv ("
			  << points.size() << " points)" << std::endl;
	return clusters;
}

}  // namespace


int main( int argc, char** argv ) {
	// --- Well-separated 2D blobs: structural assertion ---------------------
	{
		const std::vector<std::array<double, 2>> centers = { { 0, 0 }, { 60, 0 }, { 30, 50 } };
		const auto points = optics::testdata::gaussian_blobs<double, 2>( centers, /*per_blob=*/200, /*stddev=*/1.5 );
		const auto clusters = cluster_and_export( points, /*min_pts=*/10, /*threshold=*/10.0, "blobs2d" );
		assert( count_large_clusters( clusters, /*min_size=*/50 ) == 3 );
	}

	// --- 3D blobs (e.g. a color space): export for 3D visualization --------
	{
		const std::vector<std::array<double, 3>> centers = {
			{ 0, 0, 0 }, { 60, 0, 0 }, { 0, 60, 0 }, { 0, 0, 60 } };
		const auto points = optics::testdata::gaussian_blobs<double, 3>( centers, /*per_blob=*/300, /*stddev=*/3.0 );
		const auto clusters = cluster_and_export( points, /*min_pts=*/12, /*threshold=*/15.0, "blobs3d" );
		assert( count_large_clusters( clusters, /*min_size=*/50 ) == 4 );
	}

	// --- Optional hand-drawn 2D fixture (white = empty, non-white = point) --
	if ( argc > 1 ) {
		const auto points = optics::fixture::read_ppm_points( argv[1] );
		std::cout << "  loaded " << points.size() << " points from " << argv[1] << std::endl;
		if ( !points.empty() ) {
			const std::size_t min_pts = 15;
			const double eps = 2.0 * optics::epsilon_estimation( points, min_pts );
			const auto reach_dists = optics::compute_reachability_dists( points, min_pts, eps );
			const auto clusters = optics::get_cluster_indices( reach_dists, eps );
			const auto labels = optics::io::cluster_labels( points.size(), clusters, min_pts );
			optics::io::export_points_csv( "fixture_points.csv", points, labels );
			optics::io::export_reachability_csv( "fixture_reachability.csv", reach_dists );
			std::cout << "  wrote fixture_points.csv and fixture_reachability.csv" << std::endl;
		}
	}

	std::cout << "Visual/export tests successful!" << std::endl;
	return 0;
}
