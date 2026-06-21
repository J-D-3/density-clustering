// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Diagnostic for the sparse-MST connectivity fix-up (issue #74). The k-NN-graph MST
// (detail::knn_graph_mst) builds the mutual-reachability MST from each point's graph_k nearest
// neighbours; whatever that sparse graph leaves DISCONNECTED is patched by detail::sparse_graph_mst's
// fix-up, which today joins every leftover component to point 0 at a fake uniform weight above all
// real edges. On well-separated data that is harmless (the partition is right either way). This probe
// finds the regime where it is NOT harmless: it sweeps graph_k DOWN (fragmenting the graph so the
// fix-up fires more) and reports the clustering agreement vs exact Boruvka. A collapsing rand as
// graph_k shrinks is the evidence that a smarter (FAMST-style) refinement -- find the true closest
// inter-component edge instead of anchoring to point 0 -- would let us use a smaller, faster graph_k.
//
// Emits CSV: dim,sep,n,min_samples,graph_k,knn_ms,n_clusters_ref,n_clusters,fixup_edges,rand
//   sep            : blob separation (30 = well separated, small = overlapping/hard).
//   fixup_edges    : MST edges at the inflated fix-up weight (== how often the crude patch fired).
//   rand           : Rand index of the graph_k clustering vs exact Boruvka (1.0 == identical).
//
// Usage: optics_hdbscan_fixup_probe [scale].  Release config; not a ctest.

#include <optics/hdbscan.hpp>
#include <optics/Stopwatch.hpp>
#include <optics/testdata.hpp>

#include "bench_config.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

namespace sw = stopwatch;

namespace {

double rand_index_sampled( const std::vector<int>& a, const std::vector<int>& b, std::size_t max_pts ) {
	const std::size_t n = a.size();
	const std::size_t stride = std::max<std::size_t>( 1, n / std::max<std::size_t>( 1, max_pts ) );
	std::vector<std::size_t> idx;
	for ( std::size_t i = 0; i < n; i += stride ) { idx.push_back( i ); }
	std::size_t agree = 0, total = 0;
	for ( std::size_t i = 0; i < idx.size(); ++i ) {
		for ( std::size_t j = i + 1; j < idx.size(); ++j ) {
			if ( ( a[idx[i]] == a[idx[j]] ) == ( b[idx[i]] == b[idx[j]] ) ) { ++agree; }
			++total;
		}
	}
	return total ? static_cast<double>( agree ) / static_cast<double>( total ) : 1.0;
}

int n_clusters( const std::vector<int>& labels ) {
	int mx = -1;
	for ( int l : labels ) { mx = std::max( mx, l ); }
	return mx + 1;  // labels are 0..k-1, -1 = noise
}

// Count MST edges sitting at the global-max weight -- the fix-up edges all share the single inflated
// connect weight (2*max_real + 1), so when the graph was disconnected they are exactly the top-weight
// edges. (When the graph is fully connected there are none and this just counts the single heaviest
// real edge, which the caller reads as ~0.)
std::size_t count_fixup_edges( const std::vector<optics::detail::MstEdge>& mst ) {
	double mx = 0.0;
	for ( const auto& e : mst ) { mx = std::max( mx, e.weight ); }
	std::size_t c = 0;
	for ( const auto& e : mst ) { if ( e.weight >= mx ) { ++c; } }
	return c;
}

template <std::size_t Dim>
void run( double sep, std::size_t n_blobs, std::size_t per, std::size_t mcs, std::size_t min_samples,
		  const std::vector<std::size_t>& graph_ks, unsigned nt ) {
	const auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per, sep, 1.0, 321u );
	const std::size_t n = pts.size();

	// Exact reference clustering (Boruvka == exact complete-graph MST).
	const auto ref = optics::hdbscan<double, Dim>( pts, mcs, min_samples, optics::ClusterSelectionMethod::EOM,
												   false, nt, true, {}, optics::MstAlgorithm::Boruvka );
	const int ref_k = n_clusters( ref.labels );

	for ( const std::size_t gk : graph_ks ) {
		sw::Stopwatch w;
		const auto mst = optics::detail::knn_graph_mst<double, Dim>( pts, min_samples, nt, {}, gk );
		const auto res = optics::detail::extract_from_mst( mst, n, mcs, optics::ClusterSelectionMethod::EOM, false );
		const long long ms = static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );

		const double rand = rand_index_sampled( ref.labels, res.labels, 3000 );
		std::cout << Dim << "," << sep << "," << n << "," << min_samples << "," << gk << "," << ms << ","
				  << ref_k << "," << n_clusters( res.labels ) << "," << count_fixup_edges( mst ) << ","
				  << rand << "\n";
		std::cout.flush();
	}
}

}  // namespace

int main( int argc, char** argv ) {
	const unsigned nt = bench::threads();
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }

	std::cerr << "Connectivity fix-up diagnostic (#74): sweep graph_k DOWN, watch rand vs exact Boruvka. "
			  << "min_samples=5; auto graph_k would be 10. threads=" << nt << ", scale=" << scale << ".\n";
	std::cout << "dim,sep,n,min_samples,graph_k,knn_ms,n_clusters_ref,n_clusters,fixup_edges,rand\n";

	const std::vector<std::size_t> gks = { 10, 8, 6, 5, 4, 3, 2 };
	const std::size_t per = 1500 * scale;
	// Well-separated (the easy case #73 already saw at 1.0) through overlapping (the hard case).
	for ( const double sep : { 30.0, 6.0, 3.0 } ) {
		run<8>( sep, 8, per, 15, 5, gks, nt );
		run<16>( sep, 8, per, 15, 5, gks, nt );
	}
	return 0;
}
