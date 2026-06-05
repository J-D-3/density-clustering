// Copyright Ingo Proff 2016.
// https://github.com/CrikeeIP/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Dimension-agnostic CSV export for external visualization (see tools/visualize.py).
// Optional, dependency-free, and never pulled in by the core algorithm.

#include "optics.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace optics::io {

// Per-point cluster labels derived from an index-list clustering. Clusters with
// fewer than min_cluster_size members are treated as noise (label -1); the rest
// get consecutive ids in iteration order.
inline std::vector<long long> cluster_labels( std::size_t n_points,
											   const std::vector<std::vector<std::size_t>>& clusters,
											   std::size_t min_cluster_size = 1 ) {
	std::vector<long long> labels( n_points, -1 );
	long long next_id = 0;
	for ( const auto& cluster : clusters ) {
		if ( cluster.size() < min_cluster_size ) { continue; }
		for ( const std::size_t idx : cluster ) {
			if ( idx < labels.size() ) { labels[idx] = next_id; }
		}
		++next_id;
	}
	return labels;
}

// Writes "x0,x1,...,x{Dim-1},cluster_id" rows for any dimensionality. A point's
// label is -1 (noise) when labels is empty or shorter than the cloud.
template <typename T, std::size_t Dim>
bool export_points_csv( const std::string& path, const std::vector<std::array<T, Dim>>& points,
						const std::vector<long long>& labels = {} ) {
	std::ofstream stream( path );
	if ( !stream ) { return false; }
	for ( std::size_t d = 0; d < Dim; ++d ) { stream << "x" << d << ","; }
	stream << "cluster_id\n";
	for ( std::size_t i = 0; i < points.size(); ++i ) {
		for ( std::size_t d = 0; d < Dim; ++d ) { stream << points[i][d] << ","; }
		stream << ( i < labels.size() ? labels[i] : -1 ) << "\n";
	}
	return static_cast<bool>( stream );
}

// Writes the OPTICS cluster-ordering as "order_index,point_index,reachability".
// Unreached points keep reachability -1.
inline bool export_reachability_csv( const std::string& path, const std::vector<reachability_dist>& reach_dists ) {
	std::ofstream stream( path );
	if ( !stream ) { return false; }
	stream << "order_index,point_index,reachability\n";
	for ( std::size_t i = 0; i < reach_dists.size(); ++i ) {
		stream << i << "," << reach_dists[i].point_index << "," << reach_dists[i].reach_dist << "\n";
	}
	return static_cast<bool>( stream );
}

}  // namespace optics::io
