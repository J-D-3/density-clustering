// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Tiny CSV point-cloud reader shared by the example programs (color_clustering and
// cluster_csv), so they don't each hand-roll the same parsing. Reads a numeric CSV
// with an optional `x0,x1,...` / named header into a flat row-major buffer + dimension,
// then `pack<T, Dim>` lifts it into the std::array cloud the library expects.
//
// This is example glue, not part of the library API (cf. optics/io.hpp, which is the
// public *export* side). The benchmark harnesses keep their own test/Benchmark/csv_points.hpp.

#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace example_io {

inline bool is_number( const std::string& s ) {
	if ( s.empty() ) { return false; }
	try { std::size_t n = 0; (void)std::stod( s, &n ); return n == s.size(); }
	catch ( ... ) { return false; }
}

inline std::vector<std::string> split( const std::string& line, char delim = ',' ) {
	std::vector<std::string> out;
	std::stringstream ss( line );
	std::string tok;
	while ( std::getline( ss, tok, delim ) ) { out.push_back( tok ); }
	return out;
}

// Read `path` into `flat` (row-major), setting `n` (rows) and `dim` (columns). A leading
// non-numeric row is treated as a header; its `x`-prefixed columns (else its field count)
// fix the dimension. Rows with fewer than `dim` numeric fields are skipped. Returns false
// if the file is unreadable or yields no points.
inline bool read_csv( const std::string& path, std::vector<double>& flat, std::size_t& n, std::size_t& dim ) {
	std::ifstream in( path );
	if ( !in ) { return false; }
	flat.clear();
	n = 0;
	dim = 0;
	std::string line;
	if ( std::getline( in, line ) ) {
		const auto tok = split( line );
		bool header = false;
		for ( const auto& x : tok ) { if ( !is_number( x ) ) { header = true; break; } }
		if ( header ) {
			for ( const auto& x : tok ) { if ( !x.empty() && x[0] == 'x' ) { ++dim; } }
			if ( dim == 0 ) { dim = tok.size(); }  // header without x* names
		} else {
			dim = tok.size();
			for ( std::size_t d = 0; d < dim; ++d ) { flat.push_back( std::stod( tok[d] ) ); }
			++n;
		}
	}
	while ( std::getline( in, line ) ) {
		if ( line.empty() ) { continue; }
		const auto tok = split( line );
		if ( tok.size() < dim ) { continue; }
		const std::size_t base = flat.size();
		bool ok = true;
		for ( std::size_t d = 0; d < dim; ++d ) {
			if ( !is_number( tok[d] ) ) { ok = false; break; }
			flat.push_back( std::stod( tok[d] ) );
		}
		if ( ok ) { ++n; } else { flat.resize( base ); }
	}
	return dim > 0 && n > 0;
}

// Pack the first `Dim` columns of a flat row-major buffer (stride `src_dim`) into a cloud
// of std::array<T, Dim>. `src_dim` must be >= Dim (extra trailing columns are ignored).
template <typename T, std::size_t Dim>
std::vector<std::array<T, Dim>> pack( const std::vector<double>& flat, std::size_t n, std::size_t src_dim ) {
	std::vector<std::array<T, Dim>> pts( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		for ( std::size_t d = 0; d < Dim; ++d ) { pts[i][d] = static_cast<T>( flat[i * src_dim + d] ); }
	}
	return pts;
}

}  // namespace example_io
