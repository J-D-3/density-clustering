// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Optional phase profiler for compute_reachability_dists.
//
// Define OPTICS_PROFILE at compile time (e.g. MSVC /DOPTICS_PROFILE, GCC/Clang
// -DOPTICS_PROFILE) to time the major ordering phases -- index build, neighbor
// precompute, core-distance, relax, and the ordering loop -- and print a one-line
// breakdown to stderr after each call:
//
//   [optics::profile] n=50400 index_build=12.3 ms precompute=410.0 ms core_dist=...
//
// When OPTICS_PROFILE is *not* defined every hook below is an empty inline no-op, so
// there is zero runtime overhead and -- crucially -- the call sites in optics.hpp carry
// no #ifdef of their own (the conditional compilation lives entirely in this header).

#include <cstddef>

#ifdef OPTICS_PROFILE
#include <chrono>
#include <cstdio>
#endif

namespace optics::detail {

// Per-call accumulators for the ordering phases. The member set is identical in both
// build modes so the call sites are unconditional; only the methods differ.
class PhaseProfiler {
public:
	double index_build = 0.0;
	double precompute = 0.0;
	double core_dist = 0.0;
	double relax = 0.0;
	double loop = 0.0;

#ifdef OPTICS_PROFILE
	using clock = std::chrono::steady_clock;
	using time_point = clock::time_point;

	// RAII accumulator: adds its own lifetime (ms) to `sink` on destruction. Used for
	// phases entered many times (core_dist, relax) -- one `auto _s = prof.scope(...)`.
	class Scope {
	public:
		explicit Scope( double& sink ) : sink_( sink ), t0_( clock::now() ) {}
		~Scope() { sink_ += ms_since( t0_ ); }
		Scope( const Scope& ) = delete;
		Scope& operator=( const Scope& ) = delete;

	private:
		double& sink_;
		time_point t0_;
	};

	time_point now() const { return clock::now(); }
	void add( double& sink, time_point t0 ) const { sink += ms_since( t0 ); }
	Scope scope( double& sink ) const { return Scope( sink ); }

	void report( std::size_t n ) const {
		std::fprintf( stderr,
			"[optics::profile] n=%zu  index_build=%.1f ms  precompute=%.1f ms  "
			"core_dist=%.1f ms  relax=%.1f ms  loop=%.1f ms\n",
			n, index_build, precompute, core_dist, relax, loop );
	}

private:
	static double ms_since( time_point t0 ) {
		return std::chrono::duration<double, std::milli>( clock::now() - t0 ).count();
	}

#else  // OPTICS_PROFILE undefined: every hook is an empty no-op (optimized away).
	struct time_point {};
	struct Scope {};

	time_point now() const { return {}; }
	void add( double& /*sink*/, time_point /*t0*/ ) const {}
	Scope scope( double& /*sink*/ ) const { return {}; }
	void report( std::size_t /*n*/ ) const {}
#endif
};

}  // namespace optics::detail
