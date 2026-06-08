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
#include <optics/hdbscan.hpp>
#include <optics/io.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
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

// --- cluster_threshold: flat reachability-cut -> per-point labels (-1 = noise) -
// (the paper's ExtractDBSCAN; we do not run DBSCAN). threshold < 0 => educated default.
template <std::size_t Dim>
py::array_t<long long> threshold_impl( const Array& arr, std::size_t min_pts, double threshold, double min_cluster_frac ) {
    const auto pts = to_points<Dim>( arr );
    const auto clusters = optics::cluster_threshold( pts, min_pts, threshold );
    const std::size_t min_size = std::max<std::size_t>( 1, static_cast<std::size_t>( min_cluster_frac * static_cast<double>( pts.size() ) ) );
    return to_label_array( optics::io::cluster_labels( pts.size(), clusters, min_size ) );
}

py::array_t<long long> cluster_threshold_py( const Array& arr, std::size_t min_pts, double threshold, double min_cluster_frac ) {
    switch ( check_dim( arr ) ) {
        case 1: return threshold_impl<1>( arr, min_pts, threshold, min_cluster_frac );
        case 2: return threshold_impl<2>( arr, min_pts, threshold, min_cluster_frac );
        case 3: return threshold_impl<3>( arr, min_pts, threshold, min_cluster_frac );
        case 4: return threshold_impl<4>( arr, min_pts, threshold, min_cluster_frac );
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" );
    }
}

// --- extract_xi: hierarchical (steep-area) clusters -> per-point labels --------
template <std::size_t Dim>
py::array_t<long long> xi_impl( const Array& arr, std::size_t min_pts, double chi ) {
    const auto pts = to_points<Dim>( arr );
    const auto clusters = optics::extract_xi( pts, min_pts, chi );
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

// --- hdbscan: HDBSCAN* density clustering -> labels + probabilities -----------
// Parameter-light (no epsilon/threshold): just min_cluster_size, plus an optional
// min_samples density smoother. dedup (on by default) collapses bit-identical points
// -- the big win on flat-color/quantized data. Returns labels (-1 == noise),
// per-point membership probabilities in [0, 1], and the cluster count.
template <std::size_t Dim>
py::dict hdbscan_impl( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                       const std::string& method, bool allow_single_cluster, unsigned n_threads, bool dedup ) {
    const auto pts = to_points<Dim>( arr );
    const auto sel = ( method == "leaf" || method == "Leaf" ) ? optics::ClusterSelectionMethod::Leaf
                                                              : optics::ClusterSelectionMethod::EOM;
    const auto res = optics::hdbscan( pts, min_cluster_size, min_samples, sel, allow_single_cluster, n_threads, dedup );
    py::array_t<long long> labels( static_cast<py::ssize_t>( res.labels.size() ) );
    py::array_t<double> probs( static_cast<py::ssize_t>( res.probabilities.size() ) );
    auto l = labels.mutable_unchecked<1>();
    auto p = probs.mutable_unchecked<1>();
    for ( std::size_t i = 0; i < res.labels.size(); ++i ) { l( static_cast<py::ssize_t>( i ) ) = res.labels[i]; }
    for ( std::size_t i = 0; i < res.probabilities.size(); ++i ) { p( static_cast<py::ssize_t>( i ) ) = res.probabilities[i]; }
    py::dict d;
    d["labels"] = labels;              // per point: cluster id 0..n_clusters-1, or -1 for noise
    d["probabilities"] = probs;        // per point: membership strength in [0, 1] (0 for noise)
    d["n_clusters"] = static_cast<long long>( res.n_clusters );
    return d;
}

py::dict hdbscan_py( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                     const std::string& method, bool allow_single_cluster, unsigned n_threads, bool dedup ) {
    switch ( check_dim( arr ) ) {
        case 1: return hdbscan_impl<1>( arr, min_cluster_size, min_samples, method, allow_single_cluster, n_threads, dedup );
        case 2: return hdbscan_impl<2>( arr, min_cluster_size, min_samples, method, allow_single_cluster, n_threads, dedup );
        case 3: return hdbscan_impl<3>( arr, min_cluster_size, min_samples, method, allow_single_cluster, n_threads, dedup );
        case 4: return hdbscan_impl<4>( arr, min_cluster_size, min_samples, method, allow_single_cluster, n_threads, dedup );
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" );
    }
}

optics::Metric parse_metric( const std::string& s ) {
    if ( s == "l2" || s == "L2" ) { return optics::Metric::L2; }
    if ( s == "l1" || s == "L1" ) { return optics::Metric::L1; }
    return optics::Metric::Cosine;
}

// --- shdbscan: scalable approximate HDBSCAN* (CEOs random-projection MST) ----------
// Same labels/probabilities/n_clusters shape as hdbscan(), but approximate and
// deterministic in `seed`; cosine metric by default (brightness-invariant), L2/L1 via
// random-Fourier features. dedup is on (collapses bit-identical points).
template <std::size_t Dim>
py::dict shdbscan_impl( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                        const std::string& method, unsigned seed, const std::string& metric,
                        unsigned n_threads ) {
    const auto pts = to_points<Dim>( arr );
    const auto sel = ( method == "leaf" || method == "Leaf" ) ? optics::ClusterSelectionMethod::Leaf
                                                              : optics::ClusterSelectionMethod::EOM;
    const auto res = optics::shdbscan( pts, min_cluster_size, min_samples, /*epsilon*/ -1.0,
                                       /*n_projections*/ 1024u, /*k*/ 0u, /*m*/ std::size_t( 0 ),
                                       seed, n_threads, sel, /*allow_single_cluster*/ false,
                                       parse_metric( metric ) );
    py::array_t<long long> labels( static_cast<py::ssize_t>( res.labels.size() ) );
    py::array_t<double> probs( static_cast<py::ssize_t>( res.probabilities.size() ) );
    auto l = labels.mutable_unchecked<1>();
    auto p = probs.mutable_unchecked<1>();
    for ( std::size_t i = 0; i < res.labels.size(); ++i ) { l( static_cast<py::ssize_t>( i ) ) = res.labels[i]; }
    for ( std::size_t i = 0; i < res.probabilities.size(); ++i ) { p( static_cast<py::ssize_t>( i ) ) = res.probabilities[i]; }
    py::dict d;
    d["labels"] = labels;
    d["probabilities"] = probs;
    d["n_clusters"] = static_cast<long long>( res.n_clusters );
    return d;
}

py::dict shdbscan_py( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                      const std::string& method, unsigned seed, const std::string& metric, unsigned n_threads ) {
    switch ( check_dim( arr ) ) {
        case 1: return shdbscan_impl<1>( arr, min_cluster_size, min_samples, method, seed, metric, n_threads );
        case 2: return shdbscan_impl<2>( arr, min_cluster_size, min_samples, method, seed, metric, n_threads );
        case 3: return shdbscan_impl<3>( arr, min_cluster_size, min_samples, method, seed, metric, n_threads );
        case 4: return shdbscan_impl<4>( arr, min_cluster_size, min_samples, method, seed, metric, n_threads );
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" );
    }
}

// --- soptics: scalable approximate OPTICS (CEOs) -> per-point labels ---------------
// Computes the approximate reachability ordering, then extracts clusters by a flat
// threshold cut (extract="threshold") or the hierarchical Xi method (extract="xi"),
// dropping clusters below min_cluster_frac of the cloud as noise (-1). Deterministic
// in `seed`; cosine metric by default.
template <std::size_t Dim>
py::array_t<long long> soptics_impl( const Array& arr, std::size_t min_pts, const std::string& extract,
                                     double threshold, double chi, double epsilon, unsigned seed,
                                     const std::string& metric, double min_cluster_frac, unsigned n_threads ) {
    const auto pts = to_points<Dim>( arr );
    const auto reach = optics::compute_soptics_reachability_dists(
        pts, min_pts, epsilon, /*n_projections*/ 1024u, /*k*/ 0u, /*m*/ std::size_t( 0 ),
        seed, n_threads, parse_metric( metric ) );
    std::vector<std::vector<std::size_t>> clusters;
    if ( extract == "xi" || extract == "Xi" ) {
        const auto flat = optics::get_chi_clusters_flat( reach, chi, min_pts );
        clusters = optics::get_cluster_indices( reach, flat );
    } else {
        const double t = ( threshold < 0.0 ) ? optics::detail::default_threshold( reach ) : threshold;
        clusters = optics::get_cluster_indices( reach, t );
    }
    const std::size_t min_size =
        std::max<std::size_t>( 1, static_cast<std::size_t>( min_cluster_frac * static_cast<double>( pts.size() ) ) );
    return to_label_array( optics::io::cluster_labels( pts.size(), clusters, min_size ) );
}

py::array_t<long long> soptics_py( const Array& arr, std::size_t min_pts, const std::string& extract,
                                   double threshold, double chi, double epsilon, unsigned seed,
                                   const std::string& metric, double min_cluster_frac, unsigned n_threads ) {
    switch ( check_dim( arr ) ) {
        case 1: return soptics_impl<1>( arr, min_pts, extract, threshold, chi, epsilon, seed, metric, min_cluster_frac, n_threads );
        case 2: return soptics_impl<2>( arr, min_pts, extract, threshold, chi, epsilon, seed, metric, min_cluster_frac, n_threads );
        case 3: return soptics_impl<3>( arr, min_pts, extract, threshold, chi, epsilon, seed, metric, min_cluster_frac, n_threads );
        case 4: return soptics_impl<4>( arr, min_pts, extract, threshold, chi, epsilon, seed, metric, min_cluster_frac, n_threads );
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" );
    }
}

}  // namespace

PYBIND11_MODULE( optics_py, m ) {
    m.doc() = "OPTICS density-based clustering for 1/2/3/4-D NumPy point clouds.";

    m.def( "cluster_threshold", &cluster_threshold_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "threshold" ) = -1.0, py::arg( "min_cluster_frac" ) = 0.0,
           "Flat reachability-threshold cut (the OPTICS paper's ExtractDBSCAN -- the same clustering "
           "DBSCAN would give at eps = threshold; not a DBSCAN run). threshold < 0 picks an educated "
           "default (a high percentile of the reachabilities). Returns per-point int labels (-1 = noise)." );

    // Deprecated alias for the old name.
    m.def( "cluster_dbscan", &cluster_threshold_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "threshold" ) = -1.0, py::arg( "min_cluster_frac" ) = 0.0,
           "Deprecated alias for cluster_threshold." );

    m.def( "extract_xi", &extract_xi_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "chi" ) = 0.05,
           "Hierarchical (xi steep-area) extraction, flattened to clusters (use the C++ get_chi_clusters "
           "for the tree). Returns per-point int labels (-1 = noise)." );

    m.def( "compute_reachability", &compute_reachability_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "epsilon" ) = -1.0,
           "Raw OPTICS ordering. Returns a dict with 'point_index' and 'reachability' arrays "
           "(in cluster order; reachability -1 means UNDEFINED)." );

    m.def( "hdbscan", &hdbscan_py,
           py::arg( "points" ), py::arg( "min_cluster_size" ), py::arg( "min_samples" ) = 0,
           py::arg( "method" ) = "eom", py::arg( "allow_single_cluster" ) = false,
           py::arg( "n_threads" ) = 0, py::arg( "dedup" ) = true,
           "HDBSCAN* density clustering. Parameter-light: no epsilon/threshold -- just "
           "min_cluster_size (the smallest group called a cluster, >= 2) and an optional "
           "min_samples density smoother (0 => min_cluster_size). method is 'eom' (default, "
           "most persistent clusters) or 'leaf' (finest). dedup collapses bit-identical points "
           "(on by default; the big win on flat-color data). Returns a dict with 'labels' "
           "(per-point int, -1 = noise), 'probabilities' ([0,1] membership strength), and "
           "'n_clusters'." );

    m.def( "shdbscan", &shdbscan_py,
           py::arg( "points" ), py::arg( "min_cluster_size" ), py::arg( "min_samples" ) = 0,
           py::arg( "method" ) = "eom", py::arg( "seed" ) = 42, py::arg( "metric" ) = "cosine",
           py::arg( "n_threads" ) = 0,
           "Scalable, approximate HDBSCAN* via CEOs random projections. Same return shape as "
           "hdbscan (dict: 'labels' -1=noise, 'probabilities', 'n_clusters') but approximate and "
           "deterministic in 'seed'. metric is 'cosine' (default, brightness-invariant), 'l2', or "
           "'l1'. Dedups bit-identical points internally." );

    m.def( "soptics", &soptics_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "extract" ) = "threshold",
           py::arg( "threshold" ) = -1.0, py::arg( "chi" ) = 0.05, py::arg( "epsilon" ) = -1.0,
           py::arg( "seed" ) = 42, py::arg( "metric" ) = "cosine", py::arg( "min_cluster_frac" ) = 0.0,
           py::arg( "n_threads" ) = 0,
           "Scalable, approximate OPTICS via CEOs random projections. Computes the reachability "
           "ordering, then extracts clusters by a flat cut (extract='threshold', threshold<0 = "
           "educated default) or the hierarchical Xi method (extract='xi', steepness 'chi'). "
           "Returns per-point int labels (-1 = noise); clusters below min_cluster_frac of the cloud "
           "are dropped to noise. Deterministic in 'seed'; metric 'cosine'/'l2'/'l1'." );
}
