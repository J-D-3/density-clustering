// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Preprocessing helpers for unique-point / weighted OPTICS (issue #46).
//
// The motivation is the color use case: images contain large flat-color regions, so the same
// RGB triple repeats thousands of times. OPTICS pays for every one of them (its runtime is
// ~O(neighborhood-size) per point), yet identical points carry no new information. Collapsing
// them to UNIQUE points each carrying an integer WEIGHT (= how many originals it stands for) and
// running weight-aware OPTICS on the small unique cloud gives the same clustering far faster.
//
// Two front-ends, composable as `quantize -> deduplicate -> weighted OPTICS`:
//   - deduplicate(): LOSSLESS. Merges only bit-identical coordinates (exact for color data /
//     convert_cloud<float> of integers). Useless on continuous-sensor floats (nothing is
//     bit-identical). The weighted result is the same partition as full OPTICS.
//   - quantize(): LOSSY. Snaps coordinates to a grid first, so near-identical colors (JPEG/DCT
//     artifacts, gradients) also collapse -- far higher reduction, at the cost of perturbed
//     cluster boundaries. Feed its output into deduplicate().

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace optics {

// Result of deduplicate(): the unique points, their weights, and a map from each ORIGINAL point
// to its unique-point index (so cluster results on the unique cloud can be expanded back).
template <typename T, std::size_t Dim>
struct DedupResult {
	std::vector<std::array<T, Dim>> unique_points;
	std::vector<std::size_t> weights;             // parallel to unique_points; sums to original n
	std::vector<std::size_t> unique_of_original;   // size == original n: original idx -> unique idx
};

namespace detail {

// Hash/equality for an std::array<T,Dim> coordinate, consistent with element-wise == :
// -0.0 and +0.0 compare equal AND hash equal (we canonicalize -0.0 to +0.0); NaN never equals
// itself, so distinct NaN points stay separate (allowed: equal keys must share a hash, unequal
// keys may collide). FNV-1a over the canonicalized bit patterns.
template <typename T, std::size_t Dim>
struct ArrayHash {
	std::size_t operator()( const std::array<T, Dim>& a ) const noexcept {
		std::uint64_t h = 1469598103934665603ull;  // FNV offset basis
		for ( std::size_t i = 0; i < Dim; ++i ) {
			T v = a[i];
			if ( v == T( 0 ) ) { v = T( 0 ); }  // collapse -0.0 -> +0.0
			std::uint64_t bits = 0;
			if constexpr ( sizeof( T ) == 8 ) { bits = std::bit_cast<std::uint64_t>( static_cast<double>( v ) ); }
			else { bits = std::bit_cast<std::uint32_t>( static_cast<float>( v ) ); }
			for ( int b = 0; b < static_cast<int>( sizeof( bits ) ); ++b ) {
				h ^= ( bits >> ( b * 8 ) ) & 0xffu;
				h *= 1099511628211ull;  // FNV prime
			}
		}
		return static_cast<std::size_t>( h );
	}
};

template <typename T, std::size_t Dim>
struct ArrayEq {
	bool operator()( const std::array<T, Dim>& a, const std::array<T, Dim>& b ) const noexcept {
		for ( std::size_t i = 0; i < Dim; ++i ) { if ( !( a[i] == b[i] ) ) { return false; } }
		return true;
	}
};

}  // namespace detail


// Collapse bit-identical points to unique points with integer weights. Unique points keep
// FIRST-SEEN order, so the result is deterministic. On data with no duplicates the unique cloud
// equals the input (weights all 1) -- a cheap O(n) pass that simply finds no win.
template <typename T, std::size_t Dim>
DedupResult<T, Dim> deduplicate( const std::vector<std::array<T, Dim>>& points ) {
	DedupResult<T, Dim> r;
	r.unique_of_original.resize( points.size() );
	std::unordered_map<std::array<T, Dim>, std::size_t, detail::ArrayHash<T, Dim>, detail::ArrayEq<T, Dim>> seen;
	seen.reserve( points.size() );
	for ( std::size_t i = 0; i < points.size(); ++i ) {
		auto [it, inserted] = seen.try_emplace( points[i], r.unique_points.size() );
		if ( inserted ) {
			r.unique_points.push_back( points[i] );
			r.weights.push_back( 1 );
		} else {
			++r.weights[it->second];
		}
		r.unique_of_original[i] = it->second;
	}
	return r;
}


// Cosine-aware deduplication for sOPTICS (issue #46). sOPTICS clusters by the cosine metric (points
// are L2-normalized onto the unit sphere internally), so points that are scalar multiples of each
// other -- e.g. the same hue at different brightness, (100,50,50) and (200,100,100) -- are identical
// to it, yet plain deduplicate() keeps them apart (their raw coordinates differ). This collapses
// points by DIRECTION instead: it normalizes each point and deduplicates on the unit vector. Because
// two scalar multiples normalize to only ALMOST the same bits, pass a `quantum > 0` to bin the unit
// vector onto a grid (round each component to the nearest multiple) so near-identical directions
// merge robustly; this is lossy, which matches sOPTICS's approximate nature. quantum == 0 merges only
// directions that normalize bit-identically (exact duplicates + axis-aligned multiples). The stored
// unique_points are the first-seen ORIGINAL representatives (sOPTICS re-normalizes them anyway).
// Zero-norm points (the origin) have no direction and all collapse together.
template <typename T, std::size_t Dim>
DedupResult<T, Dim> deduplicate_cosine( const std::vector<std::array<T, Dim>>& points, double quantum = 0.0 ) {
	DedupResult<T, Dim> r;
	r.unique_of_original.resize( points.size() );
	std::unordered_map<std::array<T, Dim>, std::size_t, detail::ArrayHash<T, Dim>, detail::ArrayEq<T, Dim>> seen;
	seen.reserve( points.size() );
	for ( std::size_t i = 0; i < points.size(); ++i ) {
		double nrm_sq = 0.0;
		for ( std::size_t c = 0; c < Dim; ++c ) { const double v = static_cast<double>( points[i][c] ); nrm_sq += v * v; }
		const double nrm = std::sqrt( nrm_sq );
		std::array<T, Dim> key;
		for ( std::size_t c = 0; c < Dim; ++c ) {
			double v = ( nrm > 0.0 ) ? static_cast<double>( points[i][c] ) / nrm : static_cast<double>( points[i][c] );
			if ( quantum > 0.0 ) { v = std::round( v / quantum ) * quantum; }
			key[c] = static_cast<T>( v );
		}
		auto [it, inserted] = seen.try_emplace( key, r.unique_points.size() );
		if ( inserted ) {
			r.unique_points.push_back( points[i] );  // original representative; sOPTICS re-normalizes
			r.weights.push_back( 1 );
		} else {
			++r.weights[it->second];
		}
		r.unique_of_original[i] = it->second;
	}
	return r;
}


// Expand clusters expressed as lists of UNIQUE-point indices back to lists of ORIGINAL-point
// indices, using the unique_of_original map from deduplicate(). The union over all clusters is
// exactly {0..n-1} with no gaps or duplicates (every original belongs to its unique's cluster).
inline std::vector<std::vector<std::size_t>> expand_clusters_to_original(
	const std::vector<std::vector<std::size_t>>& unique_clusters,
	const std::vector<std::size_t>& unique_of_original ) {
	// Bucket original indices by their unique index (one pass over the originals).
	std::size_t n_unique = 0;
	for ( const std::size_t u : unique_of_original ) { n_unique = std::max( n_unique, u + 1 ); }
	std::vector<std::vector<std::size_t>> originals_of_unique( n_unique );
	for ( std::size_t o = 0; o < unique_of_original.size(); ++o ) {
		originals_of_unique[unique_of_original[o]].push_back( o );
	}
	std::vector<std::vector<std::size_t>> result;
	result.reserve( unique_clusters.size() );
	for ( const auto& cluster : unique_clusters ) {
		std::vector<std::size_t> expanded;
		std::size_t total = 0;
		for ( const std::size_t u : cluster ) { total += originals_of_unique[u].size(); }
		expanded.reserve( total );
		for ( const std::size_t u : cluster ) {
			const auto& orig = originals_of_unique[u];
			expanded.insert( expanded.end(), orig.begin(), orig.end() );
		}
		result.push_back( std::move( expanded ) );
	}
	return result;
}


// Voxel/grid quantization: snap each coordinate to the centre of its `bin`-wide cell
// (floor(x/bin)*bin + bin/2). LOSSY -- it changes coordinate values -- but it makes near-identical
// colors bit-identical so deduplicate() collapses far more of them. Cardinality and order are
// preserved (output[i] corresponds to input[i]), so labels still map 1:1 to the originals. A
// non-positive bin is a no-op (returns a copy).
template <typename T, std::size_t Dim>
std::vector<std::array<T, Dim>> quantize( const std::vector<std::array<T, Dim>>& points, double bin ) {
	std::vector<std::array<T, Dim>> out( points.size() );
	if ( bin <= 0.0 ) { return points; }
	for ( std::size_t i = 0; i < points.size(); ++i ) {
		for ( std::size_t c = 0; c < Dim; ++c ) {
			const double v = static_cast<double>( points[i][c] );
			out[i][c] = static_cast<T>( std::floor( v / bin ) * bin + bin / 2.0 );
		}
	}
	return out;
}

}  // namespace optics
