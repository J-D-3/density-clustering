// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// HDBSCAN* MST strategy probe (issue #66): exact dense-Prim vs the near-exact, sub-quadratic
// k-NN-graph MST. Both feed the identical condense/stability/extract tail, so any difference is the
// MST alone. This is the acceptance evidence for #66: it should show (a) the k-NN-graph build
// scaling sub-quadratically while dense-Prim grows ~O(n^2), and (b) the two producing the SAME
// clustering (Rand index ~1.0) on well-separated data.
//
// Three MST strategies, same extraction tail:
//   DensePrim : exact, O(n^2)            -- the baseline.
//   KnnGraph  : near-exact, sub-quadratic.
//   Boruvka   : EXACT, sub-quadratic     -- same tree as DensePrim (issue #66 Phase 2).
//
// Emits CSV to stdout:
//   n,dim,min_cluster_size,denseprim_ms,knngraph_ms,boruvka_ms,knn_speedup,bor_speedup,knn_rand,bor_rand
//
// Usage: optics_hdbscan_mst_probe [scale]   (scale multiplies per-blob counts; default 1).
// Build in a Release config; not a ctest (timings vary by machine).

#include <optics/hdbscan.hpp>
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

	sw::Stopwatch w1;
	const auto dense = optics::hdbscan<double, Dim>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM,
													 false, nt, true, {}, optics::MstAlgorithm::DensePrim );
	const long long dense_ms = static_cast<long long>( bench::ceil_ms_from_us( w1.elapsed<sw::mus>() ) );

	sw::Stopwatch w2;
	const auto knn = optics::hdbscan<double, Dim>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM,
												   false, nt, true, {}, optics::MstAlgorithm::KnnGraph );
	const long long knn_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );

	sw::Stopwatch w3;
	const auto bor = optics::hdbscan<double, Dim>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM,
												   false, nt, true, {}, optics::MstAlgorithm::Boruvka );
	const long long bor_ms = static_cast<long long>( bench::ceil_ms_from_us( w3.elapsed<sw::mus>() ) );

	const double knn_ri = rand_index_sampled( dense.labels, knn.labels, 3000 );
	const double bor_ri = rand_index_sampled( dense.labels, bor.labels, 3000 );  // exact => ~1.0
	const double knn_sp = knn_ms > 0 ? static_cast<double>( dense_ms ) / static_cast<double>( knn_ms ) : 0.0;
	const double bor_sp = bor_ms > 0 ? static_cast<double>( dense_ms ) / static_cast<double>( bor_ms ) : 0.0;

	std::cout << n << "," << Dim << "," << mcs << "," << dense_ms << "," << knn_ms << "," << bor_ms << ","
			  << knn_sp << "," << bor_sp << "," << knn_ri << "," << bor_ri << "\n";
	std::cout.flush();
}

// Systematic crossover sweep: for one dimension, time the three exact/near-exact backbones across a
// geometric n ladder and report which is FASTEST per cell -- the data the hdbscan() auto-dispatch
// (#72) needs to pick a backbone by (n, dim). Boruvka (exact) is the agreement reference; dense-Prim
// is skipped above `dense_cap` (its O(n^2) is hopeless there and would dominate the sweep's runtime).
template <std::size_t Dim>
void run_sweep( unsigned nt, const std::vector<std::size_t>& ns, std::size_t mcs, std::size_t dense_cap ) {
	for ( const std::size_t n_target : ns ) {
		const std::size_t n_blobs = 8;
		const std::size_t per = std::max<std::size_t>( 1, n_target / n_blobs );
		const auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per, 30.0, 1.0, 321u );
		const std::size_t n = pts.size();

		long long dense_ms = -1;  // -1 == skipped (above the cap)
		if ( n <= dense_cap ) {
			sw::Stopwatch w;
			(void)optics::hdbscan<double, Dim>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM, false, nt,
												true, {}, optics::MstAlgorithm::DensePrim );
			dense_ms = static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
		}
		sw::Stopwatch w2;
		const auto bor = optics::hdbscan<double, Dim>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM, false, nt,
													   true, {}, optics::MstAlgorithm::Boruvka );
		const long long bor_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );
		sw::Stopwatch w3;
		const auto knn = optics::hdbscan<double, Dim>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM, false, nt,
													   true, {}, optics::MstAlgorithm::KnnGraph );
		const long long knn_ms = static_cast<long long>( bench::ceil_ms_from_us( w3.elapsed<sw::mus>() ) );

		const double knn_rand = rand_index_sampled( bor.labels, knn.labels, 3000 );  // Boruvka is exact ref
		const char* fastest = "boruvka";
		long long best = bor_ms;
		if ( dense_ms >= 0 && dense_ms < best ) { fastest = "denseprim"; best = dense_ms; }
		if ( knn_ms < best ) { fastest = "knngraph"; best = knn_ms; }

		std::cout << n << "," << Dim << "," << mcs << "," << dense_ms << "," << bor_ms << "," << knn_ms << ","
				  << knn_rand << "," << fastest << "\n";
		std::cout.flush();
	}
}

}  // namespace

int main( int argc, char** argv ) {
	const unsigned nt = bench::threads();

	// Sweep mode: `optics_hdbscan_mst_probe sweep` -- the n x dim crossover grid for auto-dispatch (#72).
	if ( argc > 1 && std::string( argv[1] ) == "sweep" ) {
		std::cerr << "HDBSCAN* MST crossover sweep (threads=" << nt << "). fastest = backbone to auto-pick "
				  << "per (n, dim); knn_rand = KnnGraph agreement vs exact Boruvka. denseprim_ms = -1 means "
				  << "skipped (above cap; its O(n^2) is hopeless there).\n";
		std::cout << "n,dim,min_cluster_size,denseprim_ms,boruvka_ms,knngraph_ms,knn_rand,fastest\n";
		const std::vector<std::size_t> ns = { 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000 };
		const std::size_t dense_cap = 32000;  // skip dense-Prim above this (timing-prohibitive, never picked)
		run_sweep<2>( nt, ns, 15, dense_cap );
		run_sweep<3>( nt, ns, 15, dense_cap );
		run_sweep<8>( nt, ns, 15, dense_cap );
		run_sweep<12>( nt, ns, 15, dense_cap );
		run_sweep<16>( nt, ns, 15, dense_cap );
		run_sweep<24>( nt, ns, 15, dense_cap );
		run_sweep<32>( nt, ns, 15, dense_cap );
		return 0;
	}

	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }
	std::cerr << "HDBSCAN* MST: dense-Prim (exact O(n^2)) vs k-NN-graph (near-exact) vs Boruvka (exact, "
			  << "sub-quadratic). threads=" << nt << ", scale=" << scale << "; rand ~1.0 == same clustering "
			  << "(bor_rand is ~1.0 by construction -- Boruvka is exact). Pass 'sweep' for the n x dim grid.\n";
	std::cout << "n,dim,min_cluster_size,denseprim_ms,knngraph_ms,boruvka_ms,knn_speedup,bor_speedup,knn_rand,bor_rand\n";
	// Growing n shows dense-Prim's O(n^2) overtaking the k-NN-graph build; raise scale to push further.
	run<3>( 6, 300 * scale, 10, nt );
	run<3>( 8, 800 * scale, 15, nt );
	run<3>( 10, 1500 * scale, 15, nt );
	run<16>( 6, 500 * scale, 10, nt );
	return 0;
}
