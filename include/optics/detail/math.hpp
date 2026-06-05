// Copyright Ingo Proff 2016.
// https://github.com/CrikeeIP/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace optics::detail {

inline constexpr double pi = std::numbers::pi;

// Squared Euclidean distance between two points. Returned as double to avoid
// overflow/precision loss when T is a low-precision type. Avoids the sqrt of dist().
template <typename T, std::size_t Dim>
inline double square_dist(const std::array<T, Dim>& a, const std::array<T, Dim>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < Dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum;
}

// Euclidean distance between two points.
template <typename T, std::size_t Dim>
inline double dist(const std::array<T, Dim>& a, const std::array<T, Dim>& b) {
    return std::sqrt(square_dist(a, b));
}

// True iff x lies within [goal - radius, goal + radius].
inline bool in_range(double goal, double x, double radius) {
    return (goal - radius) <= x && x <= (goal + radius);
}

}  // namespace optics::detail
