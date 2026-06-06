// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Precompute vs OnDemand neighbor acquisition, on CSV point clouds. Precompute caches
// every point's eps-neighborhood up front (parallel) -- fast, but O(n * avg_nbrs)
// memory. OnDemand queries one neighborhood at a time during the (sequential) ordering
// -- O(one neighborhood) memory, so it scales to clouds whose Precompute buffer would
// not fit, at the cost of the parallel-precompute speedup.
//
// Emits CSV to stdout:
//   dataset,n,dim,avg_nbrs,precompute_gb,precompute_ms,ondemand_ms
// Precompute is skipped (precompute_ms = -1) when its projected buffer exceeds --cap GB
// (default 15) so the harness can't OOM the machine. Threads default to 4 (Precompute
// only; the OnDemand query path is inherently sequential). Build in a Release config.
//
// Usage: optics_mode_compare a.csv [b.csv ...] [min_pts] [capGB]
//   (a bare integer arg is min_pts; a "cap=NN" arg sets the Precompute memory cap)

#include <optics/optics.hpp>
#include <optics/Stopwatch.hpp>

#include "bench_config.hpp"
#include "csv_points.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace sw = stopwatch;

namespace {

template <std::size_t Dim>
double avg_neighbors( const std::vector<std::array<double, Dim>>& pts, double eps, std::size_t stride ) {
	const optics::NanoflannBackend<double, Dim> backend( pts );
	double sum = 0.0;
	std::size_t q = 0;
	std::vector<std::size_t> buf;
	for ( std::size_t i = 0; i < pts.size(); i += stride, ++q ) {
		buf.clear();
		backend.radius_search( pts[i], static_cast<double>( eps ), buf );
		sum += static_cast<double>( buf.size() );
	}
	return q ? sum / static_cast<double>( q ) : 0.0;
}

template <std::size_t Dim>
void run( const std::string& name, const std::vector<double>& flat, std::size_t n,
		  std::size_t min_pts, unsigned nt, double cap_gb ) {
	const auto pts = bench::pack<Dim>( flat, n );
	const double eps = optics::epsilon_estimation( pts, min_pts );
	const std::size_t stride = std::max<std::size_t>( 1, n / 1500 );
	const double avg = avg_neighbors<Dim>( pts, eps, stride );
	const double gb = static_cast<double>( n ) * avg * 8.0 / 1e9;

	double pre_ms = -1.0;
	if ( gb <= cap_gb ) {
		sw::Stopwatch w;
		(void)optics::compute_reachability_dists<double, Dim>(
			pts, min_pts, -1.0, optics::NeighborMode::Precompute, nt );
		pre_ms = w.elapsed<sw::ms>();
	}

	sw::Stopwatch w2;
	(void)optics::compute_reachability_dists<double, Dim>(
		pts, min_pts, -1.0, optics::NeighborMode::OnDemand, 1 );
	const double od_ms = w2.elapsed<sw::ms>();

	std::cout << name << "," << n << "," << Dim << "," << static_cast<long long>( avg ) << ","
			  << gb << "," << pre_ms << "," << od_ms << "\n";
	std::cout.flush();
}

}  // namespace

int main( int argc, char** argv ) {
	std::vector<std::string> paths;
	std::size_t min_pts = 10;
	double cap_gb = 15.0;
	for ( int i = 1; i < argc; ++i ) {
		const std::string a = argv[i];
		if ( a.rfind( "cap=", 0 ) == 0 ) { cap_gb = std::stod( a.substr( 4 ) ); }
		else if ( !a.empty() && a.find_first_not_of( "0123456789" ) == std::string::npos ) {
			min_pts = static_cast<std::size_t>( std::stoul( a ) );
		} else { paths.push_back( a ); }
	}
	if ( paths.empty() ) {
		std::cerr << "usage: optics_mode_compare a.csv [b.csv ...] [min_pts] [cap=GB]\n";
		return 2;
	}
	const unsigned nt = bench::threads();
	std::cerr << "mode compare (precompute threads=" << nt << ", min_pts=" << min_pts
			  << ", precompute cap=" << cap_gb << " GB; -1 ms = skipped to avoid OOM)\n";
	std::cout << "dataset,n,dim,avg_nbrs,precompute_gb,precompute_ms,ondemand_ms\n";
	for ( const auto& path : paths ) {
		std::vector<double> flat;
		std::size_t n = 0, dim = 0;
		if ( !bench::read_csv( path, flat, n, dim ) ) { std::cerr << "skip: " << path << "\n"; continue; }
		switch ( dim ) {
			case 2:  run<2>( path, flat, n, min_pts, nt, cap_gb ); break;
			case 3:  run<3>( path, flat, n, min_pts, nt, cap_gb ); break;
			case 4:  run<4>( path, flat, n, min_pts, nt, cap_gb ); break;
			case 16: run<16>( path, flat, n, min_pts, nt, cap_gb ); break;
			default: std::cerr << "skip (unsupported dim " << dim << "): " << path << "\n"; break;
		}
	}
	return 0;
}
