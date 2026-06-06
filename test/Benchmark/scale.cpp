// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Large-scale (1e6-1e7) 3D timing harness for the color-space-style workload.
// Build in a Release config. Usage: optics_scale [n_points]   (default 1000000)

#include <optics/optics.hpp>
#include <optics/testdata.hpp>
#include <optics/Stopwatch.hpp>

#include "bench_config.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace sw = stopwatch;

template <class T>
void run( std::size_t n_points ) {
	const std::size_t min_pts = 16;
	const auto points = optics::testdata::uniform_noise<T, 3>( n_points, 0.0, 1000.0 );
	const char* tname = ( sizeof( T ) == 4 ? "float " : "double" );

	const unsigned nt = bench::threads();
	sw::Stopwatch w;
	auto rd1 = optics::compute_reachability_dists( points, min_pts, -1.0, optics::NeighborMode::Precompute, nt );
	const auto precompute_ms = w.elapsed<sw::ms>();

	w.restart();
	auto rd2 = optics::compute_reachability_dists( points, min_pts, -1.0, optics::NeighborMode::OnDemand, 1 );
	const auto ondemand_ms = w.elapsed<sw::ms>();

	std::cout << "  3D " << tname << " n=" << points.size()
			  << "  precompute(x" << nt << ")=" << precompute_ms << " ms"
			  << "  ondemand(x1)=" << ondemand_ms << " ms"
			  << "  (ordered " << rd1.size() << ", " << rd2.size() << ")" << std::endl;
}

int main( int argc, char** argv ) {
	std::size_t n = 1000000;
	if ( argc > 1 ) { n = static_cast<std::size_t>( std::strtoull( argv[1], nullptr, 10 ) ); }
	std::cout << "OPTICS scale (threads=" << bench::threads()
			  << ", hw=" << std::max( 1u, std::thread::hardware_concurrency() ) << ")" << std::endl;
	run<float>( n );
	run<double>( n );
	return 0;
}
