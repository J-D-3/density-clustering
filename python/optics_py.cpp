// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Native pybind11 binding: OPTICS / HDBSCAN* / sHDBSCAN / sOPTICS for 1/2/3/4-D NumPy
// point clouds. Built only when OPTICS_BUILD_PYTHON=ON (needs pybind11). The C++ library
// itself has no Python dependency. The module is named `_optics` and ships inside the
// pure-Python `optics` package (see python/optics/); the high-level, OpenCV-friendly color
// API lives there. These functions are the thin, faithful low-level surface.
//
// Image note: every clustering function takes an optional `voxel` (grid size in coordinate
// units): voxel > 0 snaps colors to a grid first (optics::quantize) so near-identical colors
// merge -- the study's speed/quality knob (sweet spot ~4 in RGB units, ~2 in Lab). voxel <= 0
// is a no-op. Bit-identical points are always deduplicated internally (lossless, the big win
// on flat-color images); the per-point result is identical to clustering every pixel.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <optics/optics.hpp>
#include <optics/hdbscan.hpp>
#include <optics/io.hpp>
#include <optics/preprocess.hpp>

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

// Optional voxel pre-pass: snap to a grid so near-identical colors merge. No-op for voxel <= 0
// (optics::quantize returns a copy), so the default path is byte-for-byte unchanged.
template <std::size_t Dim>
std::vector<std::array<double, Dim>> apply_voxel( std::vector<std::array<double, Dim>> pts, double voxel ) {
    if ( voxel > 0.0 ) { return optics::quantize( pts, voxel ); }
    return pts;
}

py::array_t<long long> to_label_array( const std::vector<long long>& labels ) {
    py::array_t<long long> out( static_cast<py::ssize_t>( labels.size() ) );
    auto o = out.mutable_unchecked<1>();
    for ( std::size_t i = 0; i < labels.size(); ++i ) { o( static_cast<py::ssize_t>( i ) ) = labels[i]; }
    return out;
}

template <std::size_t Dim>
py::array_t<double> to_point_array( const std::vector<std::array<double, Dim>>& pts ) {
    py::array_t<double> out( { static_cast<py::ssize_t>( pts.size() ), static_cast<py::ssize_t>( Dim ) } );
    auto o = out.template mutable_unchecked<2>();
    for ( std::size_t i = 0; i < pts.size(); ++i ) {
        for ( std::size_t d = 0; d < Dim; ++d ) { o( static_cast<py::ssize_t>( i ), static_cast<py::ssize_t>( d ) ) = pts[i][d]; }
    }
    return out;
}

std::size_t check_dim( const Array& arr ) {
    if ( arr.ndim() != 2 ) { throw std::invalid_argument( "points must be a 2-D (N, Dim) array" ); }
    return static_cast<std::size_t>( arr.shape( 1 ) );
}

std::size_t min_size_of( double frac, std::size_t n ) {
    return std::max<std::size_t>( 1, static_cast<std::size_t>( frac * static_cast<double>( n ) ) );
}

optics::Metric parse_metric( const std::string& s ) {
    if ( s == "l2" || s == "L2" ) { return optics::Metric::L2; }
    if ( s == "l1" || s == "L1" ) { return optics::Metric::L1; }
    return optics::Metric::Cosine;
}

// Macro to dispatch a templated impl over the supported 1..4 dimensions.
#define OPTICS_DIM_DISPATCH( IMPL, ... )                                          \
    switch ( check_dim( arr ) ) {                                                 \
        case 1: return IMPL<1>( __VA_ARGS__ );                                    \
        case 2: return IMPL<2>( __VA_ARGS__ );                                    \
        case 3: return IMPL<3>( __VA_ARGS__ );                                    \
        case 4: return IMPL<4>( __VA_ARGS__ );                                    \
        default: throw std::invalid_argument( "only 1..4 dimensions are supported" ); \
    }

// --- quantize: lossy voxel snap (exposed as a standalone utility) --------------
template <std::size_t Dim>
py::array_t<double> quantize_impl( const Array& arr, double bin ) {
    return to_point_array<Dim>( optics::quantize( to_points<Dim>( arr ), bin ) );
}
py::array_t<double> quantize_py( const Array& arr, double bin ) { OPTICS_DIM_DISPATCH( quantize_impl, arr, bin ) }

// --- cluster_threshold: flat reachability-cut -> per-point labels (-1 = noise) -
template <std::size_t Dim>
py::array_t<long long> threshold_impl( const Array& arr, std::size_t min_pts, double threshold,
                                       double min_cluster_frac, double voxel ) {
    const auto pts = apply_voxel<Dim>( to_points<Dim>( arr ), voxel );
    const auto clusters = optics::cluster_threshold( pts, min_pts, threshold );
    return to_label_array( optics::io::cluster_labels( pts.size(), clusters, min_size_of( min_cluster_frac, pts.size() ) ) );
}
py::array_t<long long> cluster_threshold_py( const Array& arr, std::size_t min_pts, double threshold,
                                             double min_cluster_frac, double voxel ) {
    OPTICS_DIM_DISPATCH( threshold_impl, arr, min_pts, threshold, min_cluster_frac, voxel )
}

// --- extract_xi: hierarchical (steep-area) clusters -> per-point labels --------
template <std::size_t Dim>
py::array_t<long long> xi_impl( const Array& arr, std::size_t min_pts, double chi,
                                double min_cluster_frac, double voxel ) {
    const auto pts = apply_voxel<Dim>( to_points<Dim>( arr ), voxel );
    const auto clusters = optics::extract_xi( pts, min_pts, chi );
    return to_label_array( optics::io::cluster_labels( pts.size(), clusters, min_size_of( min_cluster_frac, pts.size() ) ) );
}
py::array_t<long long> extract_xi_py( const Array& arr, std::size_t min_pts, double chi,
                                      double min_cluster_frac, double voxel ) {
    OPTICS_DIM_DISPATCH( xi_impl, arr, min_pts, chi, min_cluster_frac, voxel )
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
    OPTICS_DIM_DISPATCH( reach_impl, arr, min_pts, epsilon )
}

py::dict hdbscan_result_to_dict( const optics::HdbscanResult& res ) {
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

// --- hdbscan: exact HDBSCAN* -> labels + probabilities ------------------------
template <std::size_t Dim>
py::dict hdbscan_impl( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                       const std::string& method, bool allow_single_cluster, unsigned n_threads,
                       bool dedup, double voxel ) {
    const auto pts = apply_voxel<Dim>( to_points<Dim>( arr ), voxel );
    const auto sel = ( method == "leaf" || method == "Leaf" ) ? optics::ClusterSelectionMethod::Leaf
                                                              : optics::ClusterSelectionMethod::EOM;
    return hdbscan_result_to_dict(
        optics::hdbscan( pts, min_cluster_size, min_samples, sel, allow_single_cluster, n_threads, dedup ) );
}
py::dict hdbscan_py( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                     const std::string& method, bool allow_single_cluster, unsigned n_threads,
                     bool dedup, double voxel ) {
    OPTICS_DIM_DISPATCH( hdbscan_impl, arr, min_cluster_size, min_samples, method, allow_single_cluster, n_threads, dedup, voxel )
}

// --- shdbscan: scalable approximate HDBSCAN* (CEOs random-projection MST) ------
template <std::size_t Dim>
py::dict shdbscan_impl( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                        const std::string& method, unsigned seed, const std::string& metric,
                        unsigned n_threads, double voxel ) {
    const auto pts = apply_voxel<Dim>( to_points<Dim>( arr ), voxel );
    const auto sel = ( method == "leaf" || method == "Leaf" ) ? optics::ClusterSelectionMethod::Leaf
                                                              : optics::ClusterSelectionMethod::EOM;
    return hdbscan_result_to_dict(
        optics::shdbscan( pts, min_cluster_size, min_samples, /*epsilon*/ -1.0, /*n_projections*/ 1024u,
                          /*k*/ 0u, /*m*/ std::size_t( 0 ), seed, n_threads, sel,
                          /*allow_single_cluster*/ false, parse_metric( metric ) ) );
}
py::dict shdbscan_py( const Array& arr, std::size_t min_cluster_size, std::size_t min_samples,
                      const std::string& method, unsigned seed, const std::string& metric,
                      unsigned n_threads, double voxel ) {
    OPTICS_DIM_DISPATCH( shdbscan_impl, arr, min_cluster_size, min_samples, method, seed, metric, n_threads, voxel )
}

// --- soptics: scalable approximate OPTICS (CEOs) -> per-point labels ----------
// Deduplicates first (weight-aware), so it is fast on dense images, then extracts by a flat
// threshold cut or the hierarchical Xi method and expands labels back to the original points.
template <std::size_t Dim>
py::array_t<long long> soptics_impl( const Array& arr, std::size_t min_pts, const std::string& extract,
                                     double threshold, double chi, double epsilon, unsigned seed,
                                     const std::string& metric, double min_cluster_frac,
                                     unsigned n_threads, double voxel ) {
    const auto pts = apply_voxel<Dim>( to_points<Dim>( arr ), voxel );
    const auto dd = optics::deduplicate( pts );
    const auto reach = optics::compute_soptics_reachability_dists(
        dd.unique_points, min_pts, epsilon, /*n_projections*/ 1024u, /*k*/ 0u, /*m*/ std::size_t( 0 ),
        seed, n_threads, parse_metric( metric ), /*kernel_scale*/ 0.0, dd.weights );
    std::vector<std::vector<std::size_t>> uclusters;
    if ( extract == "xi" || extract == "Xi" ) {
        std::vector<std::size_t> w_ord( reach.size() );
        for ( std::size_t i = 0; i < reach.size(); ++i ) { w_ord[i] = dd.weights[reach[i].point_index]; }
        const auto flat = optics::get_chi_clusters_flat( reach, chi, min_pts, 0.0, 0, w_ord );
        uclusters = optics::get_cluster_indices( reach, flat );
    } else {
        const double t = ( threshold < 0.0 ) ? optics::detail::default_threshold( reach ) : threshold;
        uclusters = optics::get_cluster_indices( reach, t );
    }
    const auto clusters = optics::expand_clusters_to_original( uclusters, dd.unique_of_original );
    return to_label_array( optics::io::cluster_labels( pts.size(), clusters, min_size_of( min_cluster_frac, pts.size() ) ) );
}
py::array_t<long long> soptics_py( const Array& arr, std::size_t min_pts, const std::string& extract,
                                   double threshold, double chi, double epsilon, unsigned seed,
                                   const std::string& metric, double min_cluster_frac,
                                   unsigned n_threads, double voxel ) {
    OPTICS_DIM_DISPATCH( soptics_impl, arr, min_pts, extract, threshold, chi, epsilon, seed, metric, min_cluster_frac, n_threads, voxel )
}

#undef OPTICS_DIM_DISPATCH

}  // namespace

PYBIND11_MODULE( _optics, m ) {
    m.doc() = "Native OPTICS / HDBSCAN* / sHDBSCAN / sOPTICS clustering for 1/2/3/4-D NumPy clouds. "
              "Use the high-level optics.cluster_image() for OpenCV color pipelines.";

    m.def( "quantize", &quantize_py, py::arg( "points" ), py::arg( "bin" ),
           "Voxel-snap each coordinate to a grid of size `bin` (lossy). Near-identical colors merge, "
           "which deduplication then collapses -- the image speed/quality knob (sweet spot ~4 in RGB "
           "units, ~2 in Lab). bin <= 0 is a no-op. Returns the snapped (N, Dim) float array." );

    m.def( "cluster_threshold", &cluster_threshold_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "threshold" ) = -1.0,
           py::arg( "min_cluster_frac" ) = 0.0, py::arg( "voxel" ) = 0.0,
           "Flat reachability-threshold cut (the OPTICS paper's ExtractDBSCAN -- the same clustering "
           "DBSCAN would give at eps = threshold; not a DBSCAN run). threshold < 0 picks an educated "
           "default (a high percentile of the reachabilities). Clusters below min_cluster_frac of the "
           "cloud become noise. `voxel` > 0 snaps colors to a grid first. Returns per-point int labels "
           "(-1 = noise)." );

    // Deprecated alias for the old name.
    m.def( "cluster_dbscan", &cluster_threshold_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "threshold" ) = -1.0,
           py::arg( "min_cluster_frac" ) = 0.0, py::arg( "voxel" ) = 0.0,
           "Deprecated alias for cluster_threshold." );

    m.def( "extract_xi", &extract_xi_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "chi" ) = 0.05,
           py::arg( "min_cluster_frac" ) = 0.0, py::arg( "voxel" ) = 0.0,
           "Hierarchical (xi steep-area) extraction, flattened to clusters (use the C++ get_chi_clusters "
           "for the tree). Clusters below min_cluster_frac of the cloud become noise. `voxel` > 0 snaps "
           "colors to a grid first. Returns per-point int labels (-1 = noise)." );

    m.def( "compute_reachability", &compute_reachability_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "epsilon" ) = -1.0,
           "Raw OPTICS ordering. Returns a dict with 'point_index' and 'reachability' arrays "
           "(in cluster order; reachability -1 means UNDEFINED)." );

    m.def( "hdbscan", &hdbscan_py,
           py::arg( "points" ), py::arg( "min_cluster_size" ), py::arg( "min_samples" ) = 0,
           py::arg( "method" ) = "eom", py::arg( "allow_single_cluster" ) = false,
           py::arg( "n_threads" ) = 0, py::arg( "dedup" ) = true, py::arg( "voxel" ) = 0.0,
           "HDBSCAN* density clustering -- the recommended color clusterer. Parameter-light: no "
           "epsilon/threshold, just min_cluster_size (the smallest group called a cluster, >= 2) and an "
           "optional min_samples density smoother (0 => min_cluster_size). method is 'eom' (default, "
           "most persistent clusters) or 'leaf' (finest). dedup collapses bit-identical points (on by "
           "default; the big win on flat-color data). `voxel` > 0 snaps colors to a grid first. Returns "
           "a dict with 'labels' (per-point int, -1 = noise), 'probabilities' ([0,1] membership "
           "strength), and 'n_clusters'." );

    m.def( "shdbscan", &shdbscan_py,
           py::arg( "points" ), py::arg( "min_cluster_size" ), py::arg( "min_samples" ) = 0,
           py::arg( "method" ) = "eom", py::arg( "seed" ) = 42, py::arg( "metric" ) = "cosine",
           py::arg( "n_threads" ) = 0, py::arg( "voxel" ) = 0.0,
           "Scalable, approximate HDBSCAN* via CEOs random projections. Same return shape as hdbscan "
           "(dict: 'labels' -1=noise, 'probabilities', 'n_clusters') but approximate and deterministic "
           "in 'seed'. For COLOR work pass metric='l2' (the default 'cosine' clusters by hue and merges "
           "black/white/gray); 'l1' is Manhattan. Reserve for very large clouds (>= ~1e5 unique colors) "
           "-- below that, exact hdbscan is both more accurate and faster. `voxel` > 0 snaps first." );

    m.def( "soptics", &soptics_py,
           py::arg( "points" ), py::arg( "min_pts" ), py::arg( "extract" ) = "threshold",
           py::arg( "threshold" ) = -1.0, py::arg( "chi" ) = 0.05, py::arg( "epsilon" ) = -1.0,
           py::arg( "seed" ) = 42, py::arg( "metric" ) = "cosine", py::arg( "min_cluster_frac" ) = 0.0,
           py::arg( "n_threads" ) = 0, py::arg( "voxel" ) = 0.0,
           "Scalable, approximate OPTICS via CEOs random projections (deduplicated, so fast on images). "
           "Computes the reachability ordering, then extracts clusters by a flat cut (extract='threshold', "
           "threshold<0 = educated default) or the hierarchical Xi method (extract='xi', steepness 'chi'). "
           "For COLOR work pass metric='l2' (not the default 'cosine'). Clusters below min_cluster_frac "
           "become noise. `voxel` > 0 snaps first. Deterministic in 'seed'. Returns per-point int labels." );
}
