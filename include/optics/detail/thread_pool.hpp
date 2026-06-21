// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace optics::detail {

// Resolve a requested thread count (0 => hardware_concurrency, clamped to >= 1).
inline unsigned resolve_thread_count(unsigned requested) {
    if (requested != 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1u : hw;
}

// Invoke body(i) for every i in [0, n), partitioned statically across n_threads
// worker threads. body must be safe to call concurrently for distinct i. With a
// single effective thread the loop runs inline (no thread spawn).
template <typename Body>
void parallel_for(unsigned n_threads, std::size_t n, Body&& body) {
    const unsigned threads = std::max(1u, std::min<unsigned>(resolve_thread_count(n_threads),
                                                             static_cast<unsigned>(n == 0 ? 1 : n)));
    if (threads <= 1 || n <= 1) {
        for (std::size_t i = 0; i < n; ++i) body(i);
        return;
    }

    const std::size_t chunk = (n + threads - 1) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    for (unsigned t = 1; t < threads; ++t) {
        const std::size_t begin = std::min(n, static_cast<std::size_t>(t) * chunk);
        const std::size_t end = std::min(n, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back([&body, begin, end] {
            for (std::size_t i = begin; i < end; ++i) body(i);
        });
    }
    // The calling thread handles the first chunk.
    const std::size_t main_end = std::min(n, chunk);
    for (std::size_t i = 0; i < main_end; ++i) body(i);

    for (auto& w : workers) w.join();
}

}  // namespace optics::detail
