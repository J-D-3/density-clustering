// Copyright Ingo Proff 2016.
// https://github.com/CrikeeIP/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Minimal binary-PPM (P6) reader for hand-authored 2D test fixtures: every
// non-white pixel becomes a point. Test support only -- not part of the library.

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace optics::fixture {

namespace detail {
// Read the next unsigned integer from a PPM header, skipping whitespace and
// '#' comment lines.
inline bool read_header_uint( std::istream& in, std::size_t& out ) {
	char c;
	while ( in.get( c ) ) {
		if ( c == '#' ) {
			std::string line;
			std::getline( in, line );
		} else if ( !std::isspace( static_cast<unsigned char>( c ) ) ) {
			in.unget();
			break;
		}
	}
	return static_cast<bool>( in >> out );
}
}  // namespace detail

// Returns the (x, y) coordinates of every non-white pixel in a P6 PPM image.
// Returns an empty cloud if the file is missing or not a P6 image.
inline std::vector<std::array<double, 2>> read_ppm_points( const std::string& path ) {
	std::vector<std::array<double, 2>> points;
	std::ifstream in( path, std::ios::binary );
	if ( !in ) { return points; }

	std::string magic;
	in >> magic;
	if ( magic != "P6" ) { return points; }

	std::size_t width = 0, height = 0, maxval = 0;
	if ( !detail::read_header_uint( in, width ) ) { return points; }
	if ( !detail::read_header_uint( in, height ) ) { return points; }
	if ( !detail::read_header_uint( in, maxval ) ) { return points; }
	if ( width == 0 || height == 0 || maxval != 255 ) { return points; }
	in.get();  // consume the single whitespace separator before the pixel data

	std::vector<unsigned char> row( width * 3 );
	for ( std::size_t y = 0; y < height; ++y ) {
		in.read( reinterpret_cast<char*>( row.data() ), static_cast<std::streamsize>( row.size() ) );
		if ( !in ) { break; }
		for ( std::size_t x = 0; x < width; ++x ) {
			const unsigned char r = row[x * 3 + 0];
			const unsigned char g = row[x * 3 + 1];
			const unsigned char b = row[x * 3 + 2];
			if ( !( r == 255 && g == 255 && b == 255 ) ) {
				points.push_back( { static_cast<double>( x ), static_cast<double>( y ) } );
			}
		}
	}
	return points;
}

}  // namespace optics::fixture
