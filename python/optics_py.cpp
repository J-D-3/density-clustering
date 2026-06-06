// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Optional pybind11 binding: expose OPTICS for 1/2/3/4-D NumPy point clouds.
// Built only when OPTICS_BUILD_PYTHON=ON (needs pybind11). The C++ library itself
// has no Python dependency.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <optics/optics.hpp>
#include <optics/io.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <std::size_t Dim>
std::vector<std::array<double, Dim>> to_points( const Array& arr ) {
    auto buf = arr.template unchecked<2>();
    const std::size_t n = static_cast<std::size_t>( buf.shape( 0 ) );
    std::vector<std::array<double, Dim>> pts( n );
    for ( std::size_t i = 0; i < n; ++i ) {
        for ( std::size_t d = 0; d < Dim; ++d ) { pts[i][d] = buf( i, static_cast<py::ssize_t>( d ) ); }
    }
    return pts;
}

py::array_t<long long> to_label_array( const std::vector<long long>& labels ) {
    py::array_t<long long> out( static_cast<py::ssize_t>( labels.size() ) );
    auto o = out.mutable_unchecked<1>();
    for ( std::size_t i = 0; i < labels.size(); ++i ) { o( static_cast<py::ssize_t>( i ) ) = labels[i]; }
    return out;
}

std::size_t check_dim( const Array& arr ) {
    if ( arr.ndim() != 2 ) { throw std::invalid_argument( "points must be a 2-D (N, Dim) array" ); }
    return static_cast<std::size_t>( arr.shape( 1 ) );
}

// --- cluster_dbscan: flat threshold cut -> per-point labels (-1 = noise) ------
template <std::size_t Dim>
py::array_t<long long> dbscan_impl( const Array& arr, std::size_t min_pts, double threshold, double min_cluster_frac ) {
    const auto pts = to_points<Dim>( arr );
    const auto clusters = optics::cluster_dbscan( pts, min_pts, threshold );
    const std::size_t min_size = std::max<std::size_t>( 1, static_cast<std::size_t>( min_cluster_frac * static_cast<double>( pts.size() ) ) );
    return to_label_array( optics::io::cluster_labels( pts.size(), clusters, min_size ) );
}

py::array_t<long long> cluster_dbscan_py( const Array& arr, std::size_t min_pts, double threshold, double min_cluster_frac ) {
    switch ( check_dim( arr ) ) {
        case 1: return dbscan_impl<1>( arr, min_pts, threshold, min_cluster_frac );
        case 2: return dbscan_impl<2>( arr, min_pts, threshold, min_cluster_frac );
        case 3: return dbscan_impl<3>( arr, min_pts, threshold, min_cluster_frac );
        case 4: return dbscan_impl<4>( arr, min_pts, threshold, min_cluster_frac );
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" );
    }
}

// --- extract_xi: hierarchical (steep-area) clusters -> per-point labels --------
template <std::size_t Dim>
py::array_t<long long> xi_impl( const Array& arr, std::size_t min_pts, double chi ) {
    const auto pts = to_points<Dim>( arr );
    const auto reach = optics::compute_reachability_dists( pts, min_pts );
    const auto clusters = optics::extract_xi( reach, chi, min_pts );
    return to_label_array( optics::io::cluster_labels( pts.size(), clusters, 1 ) );
}

py::array_t<long long> extract_xi_py( const Array& arr, std::size_t min_pts, double chi ) {
    switch ( check_dim( arr ) ) {
        case 1: return xi_impl<1>( arr, min_pts, chi );
        case 2: return xi_impl<2>( arr, min_pts, chi );
        case 3: return xi_impl<3>( arr, min_pts, chi );
        case 4: return xi_impl<4>( arr, min_pts, chi );
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" );
    }
}

// --- compute_reachability: the raw ordering + reachability --------------------
template <std::size_t Dim>
py::dict reach_impl( const Array& arr, std::size_t min_pts, double epsilon ) {
    const auto pts = to_points<Dim>( arr );
    const auto reach = optics::compute_reachability_dists( pts, min_pts, epsilon );
    py::array_t<long long> pidx( static_cast<py::ssize_t>( reach.size() ) );
    py::array_t<double> rd( static_cast<py::ssize_t>( reach.size() ) );
    auto p = pidx.mutable_unchecked<1>();
    auto r = rd.mutable_unchecked<1>();
    for ( std::size_t i = 0; i < reach.size(); ++i ) {
        p( static_cast<py::ssize_t>( i ) ) = static_cast<long long>( reach[i].point_index );
        r( static_cast<py::ssize_t>( i ) ) = reach[i].reach_dist;
    }
    py::dict d;
    d["point_index"] = pidx;       // cluster ordering: original index of each ordered point
    d["reachability"] = rd;        // reachability distance (-1 == UNDEFINED/unreached)
    return d;
}

py::dict compute_reachability_py( const Array& arr, std::size_t min_pts, double epsilon ) {
    switch ( check_dim( arr ) ) {
        case 1: return reach_impl<1>( arr, min_pts, epsilon );
        case 2: return reach_impl<2>( arr, min_pts, epsilon );
        case 3: return reach_impl<3>( arr, min_pts, epsilon );
        case 4: return reach_impl<4>( arr, min_pts, epsilon );
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" );
    }
}

}  // namespace

PYBIND11_MODULE( optics_py, m ) {
    m.doc() = "OPTICS density-based clustering for 1/2/3/4-D NumPy point clouds.";

    m.def( "cluster_dbscan", &cluster_dbscan_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "threshold" ), py::arg( "min_cluster_frac" ) = 0.0,
           "Flat reachability-threshold cut. Returns per-point int labels (-1 = noise)." );

    m.def( "extract_xi", &extract_xi_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "chi" ) = 0.05,
           "Hierarchical (xi steep-area) extraction. Returns per-point int labels (-1 = noise)." );

    m.def( "compute_reachability", &compute_reachability_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "epsilon" ) = -1.0,
           "Raw OPTICS ordering. Returns a dict with 'point_index' and 'reachability' arrays "
           "(in cluster order; reachability -1 means UNDEFINED)." );
}
