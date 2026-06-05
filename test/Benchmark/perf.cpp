// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Performance-regression harness (nanobench). Measures the hot paths and a few
// end-to-end scenarios with statistically stable medians, and renders a CSV.
// Build in a Release config. Re-run before/after each essential change and
// compare against perf/baseline.csv (see docs/ROADMAP-0.9.md).

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <optics/optics.hpp>
#include <optics/testdata.hpp>

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

template <class T, std::size_t Dim>
std::vector<std::vector<std::size_t>> precompute_neighbors(
	const std::vector<std::array<T, Dim>>& points, const optics::NanoflannBackend<T, Dim>& backend, T eps ) {
	std::vector<std::vector<std::size_t>> nbrs( points.size() );
	for ( std::size_t i = 0; i < points.size(); ++i ) { backend.radius_search( points[i], eps, nbrs[i] ); }
	return nbrs;
}

// Microbenchmark: one full sweep of compute_core_dist over every point's
// neighbor list. batch(n) normalizes the reported time to per-call.
template <class T, std::size_t Dim>
void bench_core_dist( ankerl::nanobench::Bench& bench, const std::string& name, std::size_t n_points, std::size_t min_pts ) {
	const auto points = optics::testdata::uniform_noise<T, Dim>( n_points, 0.0, 1000.0 );
	const optics::NanoflannBackend<T, Dim> backend( points );
	const T eps = static_cast<T>( optics::epsilon_estimation( points, min_pts ) );
	const auto nbrs = precompute_neighbors( points, backend, eps );

	bench.batch( points.size() ).run( name, [&] {
		double acc = 0.0;
		for ( std::size_t i = 0; i < points.size(); ++i ) {
			acc += optics::detail::compute_core_dist( points[i], points, nbrs[i], min_pts ).value_or( 0.0 );
		}
		ankerl::nanobench::doNotOptimizeAway( acc );
	} );
}

// End-to-end ordering benchmark.
template <class T, std::size_t Dim>
void bench_ordering( ankerl::nanobench::Bench& bench, const std::string& name, std::size_t n_points,
					 std::size_t min_pts, optics::NeighborMode mode, unsigned threads ) {
	const auto points = optics::testdata::uniform_noise<T, Dim>( n_points, 0.0, 1000.0 );
	bench.batch( 1 ).run( name, [&] {
		auto rd = optics::compute_reachability_dists( points, min_pts, -1.0, mode, threads );
		ankerl::nanobench::doNotOptimizeAway( rd.size() );
	} );
}

}  // namespace


int main() {
	const unsigned hw = std::max( 1u, std::thread::hardware_concurrency() );
	ankerl::nanobench::Bench bench;
	bench.title( "OPTICS perf (hw=" + std::to_string( hw ) + ")" ).warmup( 1 ).relative( false );

	// Hot path that #12 targets. Each lambda call sweeps all points (~ms), so
	// force several iterations per epoch for a stable median.
	bench.minEpochIterations( 10 );
	bench_core_dist<double, 3>( bench, "core_dist 3D double (30k)", 30000, 16 );
	bench_core_dist<float, 3>( bench, "core_dist 3D float  (30k)", 30000, 16 );
	bench_core_dist<double, 16>( bench, "core_dist 16D double (8k)", 8000, 16 );

	// End-to-end ordering across modes / threads / precision. A few iterations
	// per epoch to tame run-to-run variance (each call is ~100 ms).
	bench.minEpochIterations( 3 );
	bench_ordering<double, 3>( bench, "ordering 3D double 30k precompute x1", 30000, 16, optics::NeighborMode::Precompute, 1 );
	bench_ordering<double, 3>( bench, "ordering 3D double 30k precompute xHW", 30000, 16, optics::NeighborMode::Precompute, hw );
	bench_ordering<double, 3>( bench, "ordering 3D double 30k ondemand x1", 30000, 16, optics::NeighborMode::OnDemand, 1 );
	bench_ordering<float, 3>( bench, "ordering 3D float 30k precompute xHW", 30000, 16, optics::NeighborMode::Precompute, hw );
	bench_ordering<double, 16>( bench, "ordering 16D double 8k precompute xHW", 8000, 16, optics::NeighborMode::Precompute, hw );

	std::ofstream csv( "optics_perf.csv" );
	bench.render( ankerl::nanobench::templates::csv(), csv );
	std::cout << "\nWrote optics_perf.csv" << std::endl;
	return 0;
}
