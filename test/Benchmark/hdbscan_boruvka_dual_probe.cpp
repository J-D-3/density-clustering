// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Round-adaptive dual-tree Boruvka MST (issue #75). The per-round profile (-DOPTICS_BORUVKA_PROFILE)
// showed the LATE Boruvka rounds -- few, large components -- dominate the runtime, and those rounds
// have mostly-PURE query leaves. detail::exact_mutual_reachability_mst now switches to a leaf-batched
// dual-tree once num_components <= dual_max_components (early rounds stay per-point, where the dual-tree
// was reverted as slower). This probe measures the win: same data + same core distances, timing the
// pre-#75 per-point path (dual_max_components = 0) against the round-adaptive path (auto), and verifying
// the MST total weight is IDENTICAL (the dual-tree is exact -- it is the same minimum spanning tree).
//
// Emits CSV: n,dim,perpoint_ms,adaptive_ms,speedup,weight_rel_diff
//   speedup        : perpoint_ms / adaptive_ms (>1 == round-adaptive is faster).
//   weight_rel_diff: |sum_w(adaptive) - sum_w(perpoint)| / sum_w(perpoint); must be ~0 (exactness).
//
// Usage: optics_hdbscan_boruvka_dual_probe [scale].  Release config; not a ctest.

#include <optics/hdbscan.hpp>
#include <optics/Stopwatch.hpp>
#include <optics/testdata.hpp>

#include "bench_config.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace sw = stopwatch;

namespace {

double total_weight( const std::vector<optics::detail::BoruvkaEdge>& mst ) {
	double s = 0.0;
	for ( const auto& e : mst ) { s += e.weight; }
	return s;
}

template <std::size_t Dim>
void run( std::size_t n_blobs, std::size_t per, std::size_t min_samples, unsigned nt ) {
	const auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per, 30.0, 1.0, 321u );
	const std::size_t n = pts.size();

	// Core distances once (shared by both MST variants, so only the traversal differs).
	const auto core = optics::detail::hdbscan_core_distances<double, Dim, optics::NanoflannBackend<double, Dim>>(
		pts, min_samples, nt );

	// Baseline: pure per-point search (dual disabled = pre-#75 behaviour).
	sw::Stopwatch w1;
	const auto mst_pp = optics::detail::exact_mutual_reachability_mst<double, Dim>( pts, core, nt, 16, 0 );
	const long long pp_ms = static_cast<long long>( bench::ceil_ms_from_us( w1.elapsed<sw::mus>() ) );

	// Round-adaptive: dual-tree in late rounds (auto threshold).
	sw::Stopwatch w2;
	const auto mst_ad = optics::detail::exact_mutual_reachability_mst<double, Dim>( pts, core, nt, 16 );
	const long long ad_ms = static_cast<long long>( bench::ceil_ms_from_us( w2.elapsed<sw::mus>() ) );

	const double wp = total_weight( mst_pp );
	const double rel = wp > 0.0 ? std::abs( total_weight( mst_ad ) - wp ) / wp : 0.0;
	const double speedup = ad_ms > 0 ? static_cast<double>( pp_ms ) / static_cast<double>( ad_ms ) : 0.0;

	std::cout << n << "," << Dim << "," << pp_ms << "," << ad_ms << "," << speedup << "," << rel << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	const unsigned nt = bench::threads();
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }

	std::cerr << "Round-adaptive dual-tree Boruvka (#75): per-point vs round-adaptive MST, same core "
			  << "distances. threads=" << nt << ", scale=" << scale << ". speedup>1 == adaptive faster; "
			  << "weight_rel_diff ~0 == identical (exact) MST.\n";
	std::cout << "n,dim,perpoint_ms,adaptive_ms,speedup,weight_rel_diff\n";

	for ( const std::size_t per : { std::size_t( 4000 ), std::size_t( 8000 ) } ) {
		run<2>( 8, per * scale, 5, nt );
		run<3>( 8, per * scale, 5, nt );
		run<4>( 8, per * scale, 5, nt );
		run<5>( 8, per * scale, 5, nt );
		run<6>( 8, per * scale, 5, nt );
		run<8>( 8, per * scale, 5, nt );
	}
	return 0;
}
