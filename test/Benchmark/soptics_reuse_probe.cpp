// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// sOPTICS distance-reuse probe: compares the per-phase timing of the sOPTICS ordering WITH
// the CEOs distance reuse (ceos_neighbors fills out_sq; core-dist + relax read it) against
// the recompute path (ceos without out_sq; compute_core_dist + detail::dist recompute), in
// one binary on the same data. OPTICS_PROFILE is defined here so the per-phase breakdown
// (index_build / precompute / core_dist / relax / loop, in ms) prints to stderr after each
// run. The reuse path should shrink core_dist + relax while leaving the ordering identical;
// the win grows with dimension (the eps-filter's square_dist is O(Dim)). Build Release; not
// a ctest. Usage: optics_soptics_reuse_probe [scale]   (scale multiplies the point count).

#define OPTICS_PROFILE
#include <optics/optics.hpp>
#include <optics/testdata.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t Dim = 16;  // higher-D so the per-neighbor square_dist (and its reuse) is non-trivial

// Replicates compute_soptics_reachability_dists' L2-normalization onto the unit sphere.
std::vector<std::array<double, Dim>> normalize( std::vector<std::array<double, Dim>> pts ) {
	for ( auto& p : pts ) {
		double s = 0.0;
		for ( const double c : p ) { s += c * c; }
		s = std::sqrt( s );
		if ( s > 0.0 ) { for ( auto& c : p ) { c /= s; } }
	}
	return pts;
}

}  // namespace

int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }

	const std::size_t per = 1500 * scale;
	const auto unit = normalize( optics::testdata::make_blobs<double, Dim>( 6, per, 20.0, 1.0, 21u ) );
	const std::size_t n = unit.size();
	const std::size_t min_pts = 10;
	const double eps = 2.0;  // the sOPTICS default (keep all CEOs candidates)

	optics::detail::CeosParams params;
	params.n_projections = 1024;
	params.seed = 42;

	std::cerr << "n=" << n << " dim=" << Dim << " min_pts=" << min_pts << "\n";

	// REUSE: CEOs returns squared distances; core-dist + relax read them.
	std::vector<std::vector<double>> nsq;
	const auto nbrs_r = optics::detail::ceos_neighbors( unit, eps, min_pts, params, &nsq );
	const std::vector<double>* cur_r = nullptr;
	optics::detail::PhaseProfiler p_reuse;
	const auto reuse = optics::detail::optics_order<double, Dim>(
		unit,
		[&]( std::size_t i ) -> const std::vector<std::size_t>& { cur_r = &nsq[i]; return nbrs_r[i]; },
		[&]( std::size_t, const std::vector<std::size_t>& ) { auto _s = p_reuse.scope( p_reuse.core_dist ); return optics::detail::compute_core_dist_from_sq( *cur_r, min_pts ); },
		[&]( std::size_t, std::size_t j, std::size_t ) { return std::sqrt( ( *cur_r )[j] ); },
		p_reuse );
	std::cerr << "reuse     -> ";
	p_reuse.report( n );

	// RECOMPUTE: CEOs without out_sq; core-dist + relax recompute the distances.
	const auto nbrs_c = optics::detail::ceos_neighbors( unit, eps, min_pts, params );
	optics::detail::PhaseProfiler p_recompute;
	const auto recompute = optics::detail::optics_order<double, Dim>(
		unit,
		[&]( std::size_t i ) -> const std::vector<std::size_t>& { return nbrs_c[i]; },
		[&]( std::size_t i, const std::vector<std::size_t>& nb ) { auto _s = p_recompute.scope( p_recompute.core_dist ); return optics::detail::compute_core_dist( unit[i], unit, nb, min_pts ); },
		[&]( std::size_t i, std::size_t, std::size_t o ) { return optics::detail::dist( unit[i], unit[o] ); },
		p_recompute );
	std::cerr << "recompute -> ";
	p_recompute.report( n );

	std::cerr << "orderings identical: " << ( ( reuse == recompute ) ? "yes" : "NO -- BUG" ) << "\n";
	return 0;
}
