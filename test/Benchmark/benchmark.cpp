// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Backend / mode / threads / precision benchmark. Build in a Release config.
// Usage: optics_benchmark [scale]   (scale multiplies the point counts; default 1)

#include <optics/optics.hpp>
#include <optics/testdata.hpp>
#include <optics/Stopwatch.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace sw = stopwatch;

namespace {

template <class T, std::size_t Dim, class Backend>
void run( const std::string& label, const std::vector<std::array<T, Dim>>& points,
		  std::size_t min_pts, optics::NeighborMode mode, unsigned threads ) {
	sw::Stopwatch watch;
	const auto reach = optics::compute_reachability_dists<T, Dim, Backend>( points, min_pts, -1.0, mode, threads );
	const auto ms = watch.elapsed<sw::ms>();
	std::cout << "    " << label << ": " << ms << " ms"
			  << "  (" << reach.size() << " pts ordered)" << std::endl;
}

template <class T, std::size_t Dim>
void scenario( const std::string& title, std::size_t n_points, std::size_t min_pts ) {
	const unsigned hw = std::max( 1u, std::thread::hardware_concurrency() );
	// Uniform points: the auto-estimated epsilon then yields realistic, ~min_pts
	// sized neighborhoods (rather than whole-blob neighborhoods), so the timings
	// reflect index + query cost rather than a degenerate O(n^2) blow-up.
	const auto points = optics::testdata::uniform_noise<T, Dim>( n_points, 0.0, 1000.0 );

	std::cout << title << "  (" << points.size() << " points, dim=" << Dim
			  << ", T=" << ( sizeof( T ) == 4 ? "float" : "double" ) << ", min_pts=" << min_pts << ")" << std::endl;
	run<T, Dim, optics::NanoflannBackend<T, Dim>>( "nanoflann  precompute x1 ", points, min_pts, optics::NeighborMode::Precompute, 1 );
	run<T, Dim, optics::NanoflannBackend<T, Dim>>( "nanoflann  precompute xHW", points, min_pts, optics::NeighborMode::Precompute, hw );
	run<T, Dim, optics::NanoflannBackend<T, Dim>>( "nanoflann  on-demand  x1 ", points, min_pts, optics::NeighborMode::OnDemand, 1 );
#ifdef OPTICS_ENABLE_BOOST_RTREE
	run<T, Dim, optics::BoostRTreeBackend<T, Dim>>( "boost-rtree precompute xHW", points, min_pts, optics::NeighborMode::Precompute, hw );
#endif
	std::cout << std::endl;
}

}  // namespace


int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = static_cast<std::size_t>( std::max( 1, std::atoi( argv[1] ) ) ); }

	const unsigned hw = std::max( 1u, std::thread::hardware_concurrency() );
	std::cout << "OPTICS benchmark (hardware_concurrency=" << hw << ", scale=" << scale << ")\n" << std::endl;

	// Color-space-like: 3-dim. Real targets are 1e6-1e7; scale up via the argument.
	scenario<float, 3>( "[3D float]", 100000 * scale, 16 );
	scenario<double, 3>( "[3D double]", 100000 * scale, 16 );

	// Perspective-transform-like: 16-dim (nearest-neighbor is much harder here,
	// so the default size is smaller; scale up toward 1e5-1e6 via the argument).
	scenario<float, 16>( "[16D float]", 20000 * scale, 16 );
	scenario<double, 16>( "[16D double]", 20000 * scale, 16 );

	return 0;
}
