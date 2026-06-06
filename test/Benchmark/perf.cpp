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

#include "bench_config.hpp"

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

// End-to-end ordering over a caller-supplied cloud, with an explicit CoreDistMode.
// Used for the dense-neighborhood comparison (Scan vs Knn, issue #24).
template <class T, std::size_t Dim>
void bench_ordering_mode( ankerl::nanobench::Bench& bench, const std::string& name,
						  const std::vector<std::array<T, Dim>>& points, std::size_t min_pts,
						  optics::NeighborMode mode, unsigned threads, optics::CoreDistMode cmode ) {
	bench.batch( 1 ).run( name, [&] {
		auto rd = optics::compute_reachability_dists( points, min_pts, -1.0, mode, threads, cmode );
		ankerl::nanobench::doNotOptimizeAway( rd.size() );
	} );
}

// End-to-end ordering parameterized on the neighbor-search Backend, for comparing
// backends (nanoflann exact / approximate / Boost) on the same cloud (issue #26).
template <class Backend, class T, std::size_t Dim>
void bench_ordering_backend( ankerl::nanobench::Bench& bench, const std::string& name,
							 const std::vector<std::array<T, Dim>>& points, std::size_t min_pts, unsigned threads ) {
	bench.batch( 1 ).run( name, [&] {
		auto rd = optics::compute_reachability_dists<T, Dim, Backend>(
			points, min_pts, -1.0, optics::NeighborMode::Precompute, threads );
		ankerl::nanobench::doNotOptimizeAway( rd.size() );
	} );
}

// A dense-neighborhood ("flat-color"-like) cloud: a few very tight, very populous
// blobs, so each point's eps-neighborhood holds a large fraction of the cloud --
// the regime where the Knn core-distance avoids an expensive neighborhood scan.
template <class T, std::size_t Dim>
std::vector<std::array<T, Dim>> dense_cloud( std::size_t n_points ) {
	return optics::testdata::make_blobs<T, Dim>( 3, n_points / 3, 50.0, 1.0, 1234 );
}

}  // namespace


int main() {
	const unsigned hw = std::max( 1u, std::thread::hardware_concurrency() );
	const unsigned nt = bench::threads();  // default 4, override via OPTICS_BENCH_THREADS
	const std::string xt = "x" + std::to_string( nt );
	ankerl::nanobench::Bench bench;
	bench.title( "OPTICS perf (threads=" + std::to_string( nt ) + ", hw=" + std::to_string( hw ) + ")" ).warmup( 1 ).relative( false );

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
	bench_ordering<double, 3>( bench, "ordering 3D double 30k precompute " + xt, 30000, 16, optics::NeighborMode::Precompute, nt );
	bench_ordering<double, 3>( bench, "ordering 3D double 30k ondemand x1", 30000, 16, optics::NeighborMode::OnDemand, 1 );
	bench_ordering<float, 3>( bench, "ordering 3D float 30k precompute " + xt, 30000, 16, optics::NeighborMode::Precompute, nt );
	bench_ordering<double, 16>( bench, "ordering 16D double 8k precompute " + xt, 8000, 16, optics::NeighborMode::Precompute, nt );

	// Dense-neighborhood regime (issue #24): Scan vs Knn core-distance on a cloud
	// whose eps-neighborhoods are huge. Knn should win as the neighborhoods grow.
	{
		const auto dense = dense_cloud<double, 3>( 30000 );
		bench_ordering_mode<double, 3>( bench, "dense 3D 30k core-dist scan", dense, 16, optics::NeighborMode::Precompute, nt, optics::CoreDistMode::Scan );
		bench_ordering_mode<double, 3>( bench, "dense 3D 30k core-dist knn", dense, 16, optics::NeighborMode::Precompute, nt, optics::CoreDistMode::Knn );
	}

	// Backend comparison (issue #26): same 16-D cloud, exact vs approximate nanoflann
	// (and Boost when the optional backend is enabled).
	{
		const auto pts16 = optics::testdata::uniform_noise<double, 16>( 8000, 0.0, 1000.0 );
		bench_ordering_backend<optics::NanoflannBackend<double, 16>, double, 16>( bench, "backend 16D 8k nanoflann exact", pts16, 16, nt );
		bench_ordering_backend<optics::ApproxNanoflannBackend<double, 16>, double, 16>( bench, "backend 16D 8k nanoflann approx", pts16, 16, nt );
#ifdef OPTICS_ENABLE_BOOST_RTREE
		bench_ordering_backend<optics::BoostRTreeBackend<double, 16>, double, 16>( bench, "backend 16D 8k boost rtree", pts16, 16, nt );
#endif
	}

	std::ofstream csv( "optics_perf.csv" );
	bench.render( ankerl::nanobench::templates::csv(), csv );
	std::cout << "\nWrote optics_perf.csv" << std::endl;
	return 0;
}
