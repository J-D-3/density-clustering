// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Deterministic, N-dimensional synthetic point clouds shared by the visual tests
// and the benchmarks. Header-only, dependency-free.

#include <array>
#include <cstddef>
#include <random>
#include <vector>

namespace optics::testdata {

// Gaussian blobs around the given centers (any dimension). Deterministic in seed.
template <typename T, std::size_t Dim>
std::vector<std::array<T, Dim>> gaussian_blobs( const std::vector<std::array<T, Dim>>& centers,
												std::size_t points_per_blob, double stddev,
												unsigned seed = 42 ) {
	std::mt19937 gen( seed );
	std::normal_distribution<double> jitter( 0.0, stddev );
	std::vector<std::array<T, Dim>> points;
	points.reserve( centers.size() * points_per_blob );
	for ( const auto& c : centers ) {
		for ( std::size_t i = 0; i < points_per_blob; ++i ) {
			std::array<T, Dim> p;
			for ( std::size_t k = 0; k < Dim; ++k ) {
				p[k] = static_cast<T>( static_cast<double>( c[k] ) + jitter( gen ) );
			}
			points.push_back( p );
		}
	}
	return points;
}

// Uniformly distributed noise points in the hypercube [lo, hi]^Dim.
template <typename T, std::size_t Dim>
std::vector<std::array<T, Dim>> uniform_noise( std::size_t count, double lo, double hi, unsigned seed = 7 ) {
	std::mt19937 gen( seed );
	std::uniform_real_distribution<double> u( lo, hi );
	std::vector<std::array<T, Dim>> points;
	points.reserve( count );
	for ( std::size_t i = 0; i < count; ++i ) {
		std::array<T, Dim> p;
		for ( std::size_t k = 0; k < Dim; ++k ) { p[k] = static_cast<T>( u( gen ) ); }
		points.push_back( p );
	}
	return points;
}

// Convenience: n_blobs well-separated gaussian blobs (centers drawn from a wide
// hypercube of half-width spread), each with points_per_blob points.
template <typename T, std::size_t Dim>
std::vector<std::array<T, Dim>> make_blobs( std::size_t n_blobs, std::size_t points_per_blob,
											double spread = 50.0, double stddev = 2.0, unsigned seed = 42 ) {
	std::mt19937 gen( seed );
	std::uniform_real_distribution<double> center( -spread, spread );
	std::vector<std::array<T, Dim>> centers;
	centers.reserve( n_blobs );
	for ( std::size_t b = 0; b < n_blobs; ++b ) {
		std::array<T, Dim> c;
		for ( std::size_t k = 0; k < Dim; ++k ) { c[k] = static_cast<T>( center( gen ) ); }
		centers.push_back( c );
	}
	return gaussian_blobs<T, Dim>( centers, points_per_blob, stddev, seed + 1 );
}

}  // namespace optics::testdata
