// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// HNSW backend recall-vs-speed probe (issue #47). Compares the approximate HNSW backend
// against exact nanoflann and the eps-approximate nanoflann backend in high dimensions,
// reporting index-build time, total radius-query time over a sample, and neighbor-set
// recall vs exact. HNSW's query cost is largely dimension-independent, so its advantage
// grows with Dim while its recall stays high for well-separated data. Built only when
// OPTICS_ENABLE_HNSW is ON; Release config; not a ctest.
//
// Usage: optics_hnsw_probe [scale]   (scale multiplies the point count; default 1)

#include <optics/optics.hpp>
#include <optics/Stopwatch.hpp>
#include <optics/testdata.hpp>

#include "bench_config.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace sw = stopwatch;

namespace {

// Total radius-query time (ms) over a strided sample, and recall of the returned neighbor
// sets vs an exact reference, for any backend modeling radius_search.
template <std::size_t Dim, class Backend>
std::pair<long long, double> query_and_recall(
		const Backend& backend, const std::vector<std::array<double, Dim>>& pts, double r,
		const std::vector<std::vector<std::size_t>>& truth, std::size_t stride ) {
	sw::Stopwatch w;
	std::size_t total = 0, found = 0, ti = 0;
	std::vector<std::size_t> buf;
	for ( std::size_t i = 0; i < pts.size(); i += stride, ++ti ) {
		buf.clear();
		backend.radius_search( pts[i], r, buf );
		const std::set<std::size_t> got( buf.begin(), buf.end() );
		for ( const std::size_t t : truth[ti] ) { total++; if ( got.count( t ) ) { found++; } }
	}
	const long long ms = static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
	return { ms, total ? static_cast<double>( found ) / static_cast<double>( total ) : 1.0 };
}

template <std::size_t Dim>
void run( std::size_t n_blobs, std::size_t per, double r ) {
	const auto pts = optics::testdata::make_blobs<double, Dim>( n_blobs, per, 30.0, 1.0, 71u );
	const std::size_t n = pts.size();
	const std::size_t stride = std::max<std::size_t>( 1, n / 800 );

	// Exact reference neighbor sets (the recall denominator).
	const optics::NanoflannBackend<double, Dim> exact( pts );
	std::vector<std::vector<std::size_t>> truth;
	for ( std::size_t i = 0; i < n; i += stride ) {
		truth.emplace_back();
		exact.radius_search( pts[i], r, truth.back() );
	}

	const auto [ex_ms, ex_rec] = query_and_recall<Dim>( exact, pts, r, truth, stride );

	sw::Stopwatch wb_ap;
	const optics::ApproxNanoflannBackend<double, Dim> approx( pts );
	const long long ap_build = static_cast<long long>( bench::ceil_ms_from_us( wb_ap.elapsed<sw::mus>() ) );
	const auto [ap_ms, ap_rec] = query_and_recall<Dim>( approx, pts, r, truth, stride );

	sw::Stopwatch wb_hn;
	const optics::HnswBackend<double, Dim> hnsw( pts, 32, 400 );
	const long long hn_build = static_cast<long long>( bench::ceil_ms_from_us( wb_hn.elapsed<sw::mus>() ) );
	const auto [hn_ms, hn_rec] = query_and_recall<Dim>( hnsw, pts, r, truth, stride );

	std::cout << "exact,"  << n << "," << Dim << ",-,"        << ex_ms << "," << ex_rec << "\n";
	std::cout << "approx," << n << "," << Dim << "," << ap_build << "," << ap_ms << "," << ap_rec << "\n";
	std::cout << "hnsw,"   << n << "," << Dim << "," << hn_build << "," << hn_ms << "," << hn_rec << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }
	std::cerr << "HNSW recall-vs-speed probe (scale=" << scale << ", recall vs exact nanoflann)\n";
	std::cout << "backend,n,dim,build_ms,query_ms,recall\n";
	run<8>( 6, 500 * scale, 6.0 );
	run<16>( 6, 500 * scale, 6.0 );
	run<32>( 6, 500 * scale, 7.0 );
	return 0;
}
