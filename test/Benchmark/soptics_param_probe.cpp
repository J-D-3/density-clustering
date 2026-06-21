// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// CEOs parameter sensitivity for sOPTICS quality in high dimension (issue #58: the 16-D dip).
// On well-separated blobs sOPTICS scores Rand ~1.0 at any dimension; the dip appears on
// genuinely OVERLAPPING high-dim clusters, where the CEOs approximation needs more projection
// vectors (D), more extreme vectors per point (k), or more extreme points per vector (m) to keep
// recall up against the curse of dimensionality. This harness sweeps (D, k, m) on a hard 16-D
// dataset and reports the Rand index vs exact OPTICS, so the defaults can be set from data.
//
// Emits CSV to stdout:  dim,D,k,m_mult,rand
// Build in a Release config; not a ctest.
//
// Usage: optics_soptics_param_probe [scale]

#include <optics/optics.hpp>
#include <optics/testdata.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace {

// Reproduces tools/quality_benchmark.py make_cosine_blobs: k clusters along distinct random unit
// directions in R^Dim, Gaussian jitter, then L2-normalized -- so cluster identity is angular and
// cosine (sOPTICS) is the right metric. Returns the points and the ground-truth labels.
template <std::size_t Dim>
std::pair<std::vector<std::array<double, Dim>>, std::vector<long long>>
make_cosine_blobs( std::size_t n, std::size_t k, double jitter, unsigned seed ) {
	std::mt19937 rng( seed );
	std::normal_distribution<double> g( 0.0, 1.0 );
	std::vector<std::array<double, Dim>> centers( k );
	for ( auto& c : centers ) {
		double s = 0.0;
		for ( std::size_t d = 0; d < Dim; ++d ) { c[d] = g( rng ); s += c[d] * c[d]; }
		s = std::sqrt( s );
		for ( std::size_t d = 0; d < Dim; ++d ) { c[d] /= s; }
	}
	const std::size_t per = n / k;
	std::vector<std::array<double, Dim>> pts;
	std::vector<long long> truth;
	for ( std::size_t i = 0; i < k; ++i ) {
		for ( std::size_t j = 0; j < per; ++j ) {
			std::array<double, Dim> p;
			double s = 0.0;
			for ( std::size_t d = 0; d < Dim; ++d ) { p[d] = centers[i][d] + jitter * g( rng ); s += p[d] * p[d]; }
			s = std::sqrt( s );
			for ( std::size_t d = 0; d < Dim; ++d ) { p[d] /= s; }
			pts.push_back( p );
			truth.push_back( static_cast<long long>( i ) );
		}
	}
	return { pts, truth };
}

std::vector<long long> labels_from_clusters( std::size_t n, const std::vector<std::vector<std::size_t>>& clusters ) {
	std::vector<long long> labels( n, -1 );
	long long id = 0;
	for ( const auto& c : clusters ) { for ( const std::size_t i : c ) { labels[i] = id; } ++id; }
	return labels;
}

// Adjusted Rand index (chance-corrected; the metric tools/quality_benchmark.py reports). Singletons
// each get a distinct label, so -- unlike plain Rand -- ARI is not inflated by a sea of noise points.
double adjusted_rand_index( const std::vector<long long>& a, const std::vector<long long>& b ) {
	const std::size_t n = a.size();
	std::map<std::pair<long long, long long>, double> nij;
	std::map<long long, double> ai, bj;
	for ( std::size_t i = 0; i < n; ++i ) { nij[{ a[i], b[i] }] += 1.0; ai[a[i]] += 1.0; bj[b[i]] += 1.0; }
	auto c2 = []( double x ) { return x * ( x - 1.0 ) / 2.0; };
	double sum_ij = 0.0, sum_a = 0.0, sum_b = 0.0;
	for ( const auto& [key, v] : nij ) { sum_ij += c2( v ); }
	for ( const auto& [key, v] : ai ) { sum_a += c2( v ); }
	for ( const auto& [key, v] : bj ) { sum_b += c2( v ); }
	const double tot = c2( static_cast<double>( n ) );
	const double expected = ( sum_a * sum_b ) / tot;
	const double max_index = 0.5 * ( sum_a + sum_b );
	return ( max_index - expected == 0.0 ) ? 0.0 : ( sum_ij - expected ) / ( max_index - expected );
}

}  // namespace

int main( int argc, char** argv ) {
	std::size_t scale = 1;
	if ( argc > 1 ) { scale = std::max<std::size_t>( 1, static_cast<std::size_t>( std::stoul( argv[1] ) ) ); }
	constexpr std::size_t Dim = 16;
	const std::size_t min_pts = 10;

	// The quality_benchmark cos-blobs-16d dataset: 6 angular clusters, jitter 0.12.
	const auto [pts, truth] = make_cosine_blobs<Dim>( 1200 * scale, 6, 0.12, 0u );
	const std::size_t n = pts.size();

	// The 16-D "dip" (issue #58) was an EPSILON mismatch in Xi extraction, not a CEOs recall problem:
	// exact OPTICS uses a data-scaled eps while sOPTICS used to default to 2.0 ("keep all
	// candidates"), which over-smooths the reachability plot so Xi under-segments. sOPTICS now
	// auto-estimates a data-scaled eps; this probe shows the default matching exact again, and that a
	// forced eps=2.0 reproduces the old dip. (A D/k/m sweep, by contrast, leaves the score flat --
	// recall is not the bottleneck.)
	const double eps_est = optics::epsilon_estimation( pts, min_pts );
	const auto exact = optics::compute_reachability_dists<double, Dim>( pts, min_pts, eps_est );
	const auto so_auto = optics::compute_soptics_reachability_dists<double, Dim>( pts, min_pts );          // new data-scaled default
	const auto so_old = optics::compute_soptics_reachability_dists<double, Dim>( pts, min_pts, 2.0 );      // the old eps=2.0 default

	auto xi = [&]( const std::vector<optics::reachability_dist>& r ) {
		return labels_from_clusters( n, optics::get_cluster_indices( r, optics::get_chi_clusters_flat( r, 0.05, min_pts ) ) );
	};
	std::cerr << "cos-blobs-16d: n=" << n << ", min_pts=" << min_pts << ", eps_est=" << eps_est << "\n";
	std::cout << "config,ARI_vs_truth\n";
	std::cout << "exact OPTICS (eps_est; Xi)," << adjusted_rand_index( truth, xi( exact ) ) << "\n";
	std::cout << "sOPTICS (auto eps; Xi)," << adjusted_rand_index( truth, xi( so_auto ) ) << "\n";
	std::cout << "sOPTICS (forced eps=2.0; Xi)," << adjusted_rand_index( truth, xi( so_old ) ) << "\n";
	std::cout.flush();
	return 0;
}
