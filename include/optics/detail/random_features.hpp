// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Random kernel-feature embeddings for sOPTICS (issue #51): extend the cosine CEOs
// index (detail/random_projection.hpp) to non-cosine metrics. The idea (Rahimi & Recht,
// "Random Features for Large-Scale Kernel Machines", NIPS 2007; also used by the
// sDBSCAN/sOPTICS paper, §4.4) is to map each point x -> f(x) such that the inner product
// f(x)·f(y) approximates a shift-invariant kernel K(x,y). With the cos/sin pair form the
// embedding is automatically unit-norm (||f(x)|| = 1) and f(x)·f(y) ~ K(x,y), so running
// the existing *cosine* CEOs/sOPTICS on the features orders points by the target metric:
//   - Gaussian / RBF kernel  K = exp(-||x-y||_2^2 / (2 sigma^2))  -> orders by L2.
//   - Laplacian kernel       K = exp(-||x-y||_1   / sigma)        -> orders by L1.
// Both K decrease monotonically in the corresponding distance, so cosine distance on the
// features (1 - K) increases monotonically in it -- the reachability-plot valleys match.
//
// chi^2 and Jensen-Shannon (also listed in #51) need non-negative histogram inputs and a
// different (sampling/scaling) feature construction; they are deliberately NOT implemented
// here yet -- see the issue. Reimplemented from the public method only; no third-party code.

#include "math.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace optics {

// Distance/similarity the sOPTICS clustering should reflect. Cosine is the native CEOs
// metric (no embedding); L2/L1 go through random Fourier features (detail below).
enum class Metric { Cosine, L2, L1 };

}  // namespace optics

namespace optics::detail {

// Median pairwise target-metric distance over a small stride sample -- the classic
// "median heuristic" bandwidth (sigma) so the kernel discriminates at the data's own
// scale without the caller having to tune it.
template <class T, std::size_t Dim>
double auto_kernel_scale( const std::vector<std::array<T, Dim>>& points, Metric metric ) {
	const std::size_t n = points.size();
	if ( n < 2 ) { return 1.0; }
	const std::size_t cap = std::min<std::size_t>( n, 48 );
	const std::size_t stride = std::max<std::size_t>( 1, n / cap );
	std::vector<double> dists;
	for ( std::size_t i = 0; i < n; i += stride ) {
		for ( std::size_t j = i + stride; j < n; j += stride ) {
			double s = 0.0;
			if ( metric == Metric::L1 ) {
				for ( std::size_t c = 0; c < Dim; ++c ) {
					s += std::abs( static_cast<double>( points[i][c] ) - static_cast<double>( points[j][c] ) );
				}
			} else {  // L2 (and any other non-cosine default)
				for ( std::size_t c = 0; c < Dim; ++c ) {
					const double d = static_cast<double>( points[i][c] ) - static_cast<double>( points[j][c] );
					s += d * d;
				}
				s = std::sqrt( s );
			}
			dists.push_back( s );
		}
	}
	if ( dists.empty() ) { return 1.0; }
	std::nth_element( dists.begin(), dists.begin() + dists.size() / 2, dists.end() );
	const double med = dists[dists.size() / 2];
	return med > 0.0 ? med : 1.0;
}

// Embed `points` into FeatDim random Fourier features approximating the kernel of `metric`
// at bandwidth `sigma` (see file header). FeatDim must be even: FeatDim/2 random
// frequencies, each contributing a (cos, sin) pair. The result is unit-norm per row, so
// the cosine CEOs index can be run on it directly. Deterministic in `seed`.
template <std::size_t FeatDim, class T, std::size_t Dim>
std::vector<std::array<double, FeatDim>> embed_random_features(
		const std::vector<std::array<T, Dim>>& points, Metric metric, double sigma,
		unsigned seed, unsigned n_threads = 0 ) {
	static_assert( FeatDim % 2 == 0, "embed_random_features: FeatDim must be even (cos/sin pairs)" );
	constexpr std::size_t Dp = FeatDim / 2;  // number of random frequencies
	const std::size_t n = points.size();
	std::vector<std::array<double, FeatDim>> feats( n );
	if ( n == 0 ) { return feats; }
	if ( sigma <= 0.0 ) { sigma = 1.0; }

	// Random frequencies w_j drawn from the kernel's spectral density: N(0, 1/sigma^2) for
	// the Gaussian (L2) kernel, Cauchy(0, 1/sigma) for the Laplacian (L1) kernel. The seed
	// is perturbed so these frequencies don't correlate with the CEOs projection vectors.
	std::mt19937 gen( seed ^ 0x9E3779B9u );
	std::vector<std::array<double, Dim>> W( Dp );
	if ( metric == Metric::L1 ) {
		std::cauchy_distribution<double> spectral( 0.0, 1.0 / sigma );
		for ( auto& w : W ) { for ( std::size_t c = 0; c < Dim; ++c ) { w[c] = spectral( gen ); } }
	} else {
		std::normal_distribution<double> spectral( 0.0, 1.0 / sigma );
		for ( auto& w : W ) { for ( std::size_t c = 0; c < Dim; ++c ) { w[c] = spectral( gen ); } }
	}

	const double scale = 1.0 / std::sqrt( static_cast<double>( Dp ) );
	parallel_for( n_threads, n, [&]( std::size_t i ) {
		for ( std::size_t j = 0; j < Dp; ++j ) {
			double dot = 0.0;
			for ( std::size_t c = 0; c < Dim; ++c ) { dot += static_cast<double>( points[i][c] ) * W[j][c]; }
			feats[i][2 * j]     = std::cos( dot ) * scale;
			feats[i][2 * j + 1] = std::sin( dot ) * scale;
		}
	} );
	return feats;
}

}  // namespace optics::detail
