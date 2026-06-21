// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// HDBSCAN* HNSW k-NN-graph MST: build-parameter tuning (follow-up to issue #73). The #73 probe showed
// the HNSW-backed KnnGraph backbone matches exact clustering (Rand = 1.0) but is index-build-bound, so
// it only overtakes the exact KD-tree at large n / high dim. Since the clustering only needs enough
// recall to recover the MST (not high-recall ANN), a CHEAPER index should keep Rand ~1.0 while cutting
// the dominant build cost -- moving the crossover to a smaller n. This probe quantifies that: at cells
// near the #73 crossover it compares exact-nanoflann vs default HNSW (M=16, ef_construction=200) vs the
// FastHnswBackend preset (M=8, ef_construction=48), all through the same hdbscan(..., KnnGraph).
//
// Emits CSV: n,dim,exact_ms,hnsw_ms,fast_ms,hnsw_speedup,fast_speedup,hnsw_rand,fast_rand
//   *_speedup : exact_ms / *_ms (>1 == faster than the exact KD-tree backbone).
//   *_rand    : Rand index vs the exact clustering (1.0 == identical).
//
// Usage: optics_hdbscan_hnsw_tune_probe [scale].  Built only with -DOPTICS_ENABLE_HNSW=ON; Release.

#include <optics/hdbscan.hpp>
#include <optics/hnsw_backend.hpp>
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

template <std::size_t Dim, class Backend>
std::pair<long long, std::vector<int>> time_knn( const std::vector<std::array<double, Dim>>& pts,
												 std::size_t mcs, unsigned nt ) {
	sw::Stopwatch w;
	const auto r = optics::hdbscan<double, Dim, Backend>( pts, mcs, 0, optics::ClusterSelectionMethod::EOM,
														  false, nt, true, {}, optics::MstAlgorithm::KnnGraph );
	return { static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) ), r.labels };
}

template <std::size_t Dim>
void run( std::size_t n_blobs, std::size_t per, std::size_t mcs, unsigned nt ) {
	const auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per, 30.0, 1.0, 321u );
	const std::size_t n = pts.size();

	const auto [exact_ms, exact_lab] = time_knn<Dim, optics::NanoflannBackend<double, Dim>>( pts, mcs, nt );
	const auto [hnsw_ms, hnsw_lab] = time_knn<Dim, optics::HnswBackend<double, Dim>>( pts, mcs, nt );
	const auto [fast_ms, fast_lab] = time_knn<Dim, optics::FastHnswBackend<double, Dim>>( pts, mcs, nt );

	const double hnsw_sp = hnsw_ms > 0 ? static_cast<double>( exact_ms ) / static_cast<double>( hnsw_ms ) : 0.0;
	const double fast_sp = fast_ms > 0 ? static_cast<double>( exact_ms ) / static_cast<double>( fast_ms ) : 0.0;
	const double hnsw_ri = rand_index_sampled( exact_lab, hnsw_lab, 3000 );
	const double fast_ri = rand_index_sampled( exact_lab, fast_lab, 3000 );

	std::cout << n << "," << Dim << "," << exact_ms << "," << hnsw_ms << "," << fast_ms << ","
			  << hnsw_sp << "," << fast_sp << "," << hnsw_ri << "," << fast_ri << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	const unsigned nt = bench::threads();
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }

	std::cerr << "HDBSCAN* HNSW k-NN-graph MST tuning (#73 follow-up): exact vs default HNSW (16/200) vs "
			  << "FastHnswBackend (8/48). threads=" << nt << ", scale=" << scale << ". speedup>1 == faster "
			  << "than exact; rand ~1.0 == same clustering. Goal: Fast keeps rand ~1.0 at higher speedup.\n";
	std::cout << "n,dim,exact_ms,hnsw_ms,fast_ms,hnsw_speedup,fast_speedup,hnsw_rand,fast_rand\n";

	// Cells spanning the #73 crossover (high dim, growing n), where moving the crossover left matters.
	for ( const std::size_t per : { std::size_t( 3000 ), std::size_t( 6000 ), std::size_t( 12000 ) } ) {
		run<16>( 8, per * scale, 15, nt );
		run<32>( 8, per * scale, 15, nt );
		run<64>( 8, per * scale, 15, nt );
	}
	return 0;
}
