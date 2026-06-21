// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// #55 perf probe: compares the OPTICS ordering's per-phase timing WITH the distance-reuse
// fast path (default nanoflann backend, double — reuses the squared distances the search
// already computed) against a recompute-only backend, on a dense cloud where neighborhood
// processing dominates. OPTICS_PROFILE is defined here so the per-phase breakdown
// (index_build / precompute / core_dist / relax / loop, in ms) prints to stderr after each
// run. The reuse path should shrink core_dist + relax (the phases that consume neighbor
// distances) while leaving the ordering byte-identical. Build in Release; not a ctest.
//
// Usage: optics_reuse_probe [scale]   (scale multiplies the point count; default 1)

#define OPTICS_PROFILE
#include <optics/optics.hpp>
#include <optics/testdata.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

// Forwards radius_search to nanoflann but deliberately exposes NEITHER
// radius_search_with_dists NOR knn_core_dist, so compute_reachability_dists takes the
// recompute path (Scan core-distance + detail::dist relaxation) -- the "before" baseline.
template <class T, std::size_t Dim>
class RecomputeBackend {
public:
	explicit RecomputeBackend( const std::vector<std::array<T, Dim>>& pts ) : inner_( pts ) {}
	void radius_search( const std::array<T, Dim>& p, T r, std::vector<std::size_t>& out ) const {
		inner_.radius_search( p, r, out );
	}
private:
	optics::NanoflannBackend<T, Dim> inner_;
};

int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }

	// One dense Gaussian blob in a small region: many points within eps, so the core_dist
	// scan and the relaxation (which read every neighbor's distance) dominate the runtime.
	const std::size_t n = 12000 * scale;
	const std::vector<std::array<double, 3>> centers = { { 0.0, 0.0, 0.0 } };
	const auto pts = optics::testdata::gaussian_blobs<double, 3>( centers, n, 5.0, 1u );
	const double eps = 6.0;
	const std::size_t min_pts = 10;

	std::cerr << "n=" << n << " eps=" << eps << " min_pts=" << min_pts << " (OnDemand)\n";
	std::cerr << "reuse     -> ";
	const auto reuse = optics::compute_reachability_dists<double, 3>(
		pts, min_pts, eps, optics::NeighborMode::OnDemand );
	std::cerr << "recompute -> ";
	const auto recompute = optics::compute_reachability_dists<double, 3, RecomputeBackend<double, 3>>(
		pts, min_pts, eps, optics::NeighborMode::OnDemand );

	// Sanity: the two paths must produce the identical ordering (byte-for-byte).
	std::cerr << "orderings identical: " << ( ( reuse == recompute ) ? "yes" : "NO -- BUG" ) << "\n";
	return 0;
}
