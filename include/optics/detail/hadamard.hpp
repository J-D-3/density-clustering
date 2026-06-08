// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Fast Walsh-Hadamard transform (FWHT) -- the kernel for the structured "spinner" random
// projections used by CEOs / sOPTICS (issue #58). A length-2^p vector is transformed in
// O(n log n) by the in-place butterfly below, instead of the O(n^2) of an explicit
// Hadamard matrix multiply. Combined with random sign-flips (x -> H D3 H D2 H D1 x) this
// approximates a Gaussian random projection at O(D log d) per point instead of O(D * d)
// (Ailon-Chazelle / "Fastfood" structured spinners). Dependency-free: just the standard library.
//
// The transform here is UNNORMALIZED (the natural Hadamard-ordered butterfly): applying it
// twice scales the vector by n. CEOs only uses the RANKING of projection values (each point's
// most extreme vectors, each vector's most extreme points), so the global scale is irrelevant
// and we skip the 1/sqrt(n) normalization.

#include <cstddef>
#include <vector>

namespace optics::detail {

// Smallest power of two >= x (>= 1). next_pow2(0) == 1, next_pow2(1) == 1, next_pow2(17) == 32.
inline std::size_t next_pow2( std::size_t x ) {
	std::size_t p = 1;
	while ( p < x ) { p <<= 1; }
	return p;
}

// True iff x is a power of two (and non-zero).
inline bool is_pow2( std::size_t x ) { return x != 0 && ( x & ( x - 1 ) ) == 0; }

// In-place fast Walsh-Hadamard transform over a[0..n). REQUIRES n to be a power of two
// (the caller pads to next_pow2). Unnormalized: a second application yields n * original.
inline void fwht_inplace( double* a, std::size_t n ) {
	for ( std::size_t len = 1; len < n; len <<= 1 ) {
		for ( std::size_t i = 0; i < n; i += ( len << 1 ) ) {
			for ( std::size_t j = i; j < i + len; ++j ) {
				const double u = a[j];
				const double v = a[j + len];
				a[j] = u + v;
				a[j + len] = u - v;
			}
		}
	}
}

inline void fwht_inplace( std::vector<double>& a ) { fwht_inplace( a.data(), a.size() ); }

}  // namespace optics::detail
