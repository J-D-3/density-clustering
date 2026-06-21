// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// HDBSCAN* k-NN-graph MST backbone: EXACT nanoflann k-NN vs APPROXIMATE HNSW k-NN (issue #73).
//
// MstAlgorithm::KnnGraph builds the mutual-reachability MST from each point's k-NN graph. Today that
// graph comes from the exact nanoflann KD-tree -- which `Auto` (#72) already picks at dim >= 16,
// precisely the regime where the static-Dim KD-tree prunes poorly and the exact k-NN dominates
// runtime. This probe swaps ONLY the k-NN source for the vendored approximate HNSW backend (whose
// query cost is largely dimension-independent) and measures whether that wins end-to-end, and at what
// cost to clustering agreement. Everything downstream (core distances, the sparse MST + connectivity
// fix-up, condense/stability/extract) is identical, so any difference is the neighbor graph alone.
//
// Both runs go through the same public hdbscan() with MstAlgorithm::KnnGraph; only the Backend
// template argument differs (NanoflannBackend vs HnswBackend). HNSW builds its graph index inside the
// timed call (a large fixed cost that only amortizes at larger n), so the times are honest end-to-end.
//
// `rand` is the Rand index of the HNSW clustering vs the exact-nanoflann clustering; the latter is
// itself ~1.0 vs exact Boruvka (see hdbscan_mst_probe), so rand ~1.0 here means "same clustering as
// exact". Emits CSV to stdout:
//   n,dim,min_cluster_size,exact_knn_ms,hnsw_knn_ms,speedup,rand
//
// Usage: optics_hdbscan_hnsw_mst_probe [scale]   (scale multiplies per-blob counts; default 1).
// Built only with -DOPTICS_ENABLE_HNSW=ON; Release config; not a ctest (timings vary by machine).

#include <optics/hdbscan.hpp>
#include <optics/hnsw_backend.hpp>
#include <optics/Stopwatch.hpp>
#include <optics/testdata.hpp>

#include "bench_config.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace sw = stopwatch;

namespace {

// Rand index between two label vectors over a strided subsample (keeps the O(k^2) pair loop bounded).
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

template <std::size_t Dim>
void run( std::size_t n_blobs, std::size_t per_blob, std::size_t mcs, unsigned nt ) {
	const auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per_blob, 30.0, 1.0, 321u );
	const std::size_t n = pts.size();

	// Baseline to beat: exact k-NN-graph MST via the nanoflann KD-tree (the current Auto pick at dim>=16).
	sw::Stopwatch w1;
	const auto exact = optics::hdbscan<double, Dim>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM,
													 false, nt, true, {}, optics::MstAlgorithm::KnnGraph );
	const long long exact_ms = static_cast<long long>( bench::ceil_ms_from_us( w1.elapsed<sw::mus>() ) );

	// New: the SAME k-NN-graph backbone, but the k-NN comes from the approximate HNSW index.
	sw::Stopwatch w2;
	const auto hnsw = optics::hdbscan<double, Dim, optics::HnswBackend<double, Dim>>(
		pts, mcs, 0, optics::ClusterSelectionMethod::EOM, false, nt, true, {}, optics::MstAlgorithm::KnnGraph );
	const long long hnsw_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );

	const double rand = rand_index_sampled( exact.labels, hnsw.labels, 3000 );
	const double speedup = hnsw_ms > 0 ? static_cast<double>( exact_ms ) / static_cast<double>( hnsw_ms ) : 0.0;

	std::cout << n << "," << Dim << "," << mcs << "," << exact_ms << "," << hnsw_ms << ","
			  << speedup << "," << rand << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	const unsigned nt = bench::threads();
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }

	std::cerr << "HDBSCAN* k-NN-graph MST: exact nanoflann vs approximate HNSW (issue #73). threads=" << nt
			  << ", scale=" << scale << ". speedup = exact_ms / hnsw_ms (>1 means HNSW faster); rand ~1.0 "
			  << "== same clustering as exact. HNSW index build is inside the timed call.\n";
	std::cout << "n,dim,min_cluster_size,exact_knn_ms,hnsw_knn_ms,speedup,rand\n";

	// Sweep the high-D / growing-n regime where the approximate graph is expected to pay off. Low-D rows
	// are included as the control (HNSW's fixed index-build cost should make it LOSE there).
	for ( const std::size_t per : { std::size_t( 1000 ), std::size_t( 3000 ), std::size_t( 6000 ) } ) {
		run<8>( 8, per * scale, 15, nt );
		run<16>( 8, per * scale, 15, nt );
		run<32>( 8, per * scale, 15, nt );
		run<64>( 8, per * scale, 15, nt );
	}
	return 0;
}
