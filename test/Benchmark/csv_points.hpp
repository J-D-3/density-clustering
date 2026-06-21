// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Minimal CSV point-cloud reader shared by the benchmark harnesses. Reads a numeric
// CSV with an optional `x0,x1,...` header into a flat row-major buffer + dimension.

#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace bench {

inline bool is_number( const std::string& s ) {
	if ( s.empty() ) { return false; }
	try { std::size_t n = 0; (void)std::stod( s, &n ); return n == s.size(); }
	catch ( ... ) { return false; }
}

// Read `path` into `flat` (row-major), setting `n` (rows) and `dim` (columns). A
// leading non-numeric row is treated as a header. Returns false if unreadable/empty.
inline bool read_csv( const std::string& path, std::vector<double>& flat, std::size_t& n, std::size_t& dim ) {
	std::ifstream in( path );
	if ( !in ) { return false; }
	flat.clear();
	n = 0;
	dim = 0;
	std::string line;
	auto split = []( const std::string& s ) {
		std::vector<std::string> tok;
		std::stringstream ss( s );
		std::string t;
		while ( std::getline( ss, t, ',' ) ) { tok.push_back( t ); }
		return tok;
	};
	if ( std::getline( in, line ) ) {
		const auto tok = split( line );
		bool header = false;
		for ( const auto& x : tok ) { if ( !is_number( x ) ) { header = true; break; } }
		if ( header ) {
			for ( const auto& x : tok ) { if ( !x.empty() && x[0] == 'x' ) { ++dim; } }
			if ( dim == 0 ) { dim = tok.size(); }
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

template <std::size_t Dim>
std::vector<std::array<double, Dim>> pack( const std::vector<double>& flat, std::size_t n ) {
	std::vector<std::array<double, Dim>> pts( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		for ( std::size_t d = 0; d < Dim; ++d ) { pts[i][d] = flat[i * Dim + d]; }
	}
	return pts;
}

}  // namespace bench
