// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <thread>

namespace bench {

// Runtime outputs are reported rounded UP to the next whole millisecond, so a
// sub-millisecond or fractional time never reads as a smaller (or zero) value.
inline std::uint64_t ceil_ms_from_us( std::uint64_t us ) { return ( us + 999 ) / 1000; }

// Default worker-thread count for the timing harnesses. Capped at 4 so reported
// numbers are reproducible and comparable across machines rather than scaling with
// whatever core count the dev box happens to have. Override for ad-hoc exploration
// with the OPTICS_BENCH_THREADS environment variable.
inline unsigned threads() {
    if ( const char* e = std::getenv( "OPTICS_BENCH_THREADS" ) ) {
        const int v = std::atoi( e );
        if ( v > 0 ) { return static_cast<unsigned>( v ); }
    }
    const unsigned hw = std::max( 1u, std::thread::hardware_concurrency() );
    return std::min( 4u, hw );
}

}  // namespace bench
