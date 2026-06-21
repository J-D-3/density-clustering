// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

// Per-cell executor for the 1.0.0 benchmark matrix (issue #59). Runs ONE of this library's
// algorithms (optics | soptics | hdbscan | shdbscan) on one CSV cloud, writes the predicted
// labels (in input-point order), and prints a single machine-readable RESULT line that the
// Python orchestrator (tools/run_matrix.py) parses. Generalizes optics_quality_compare +
// hdbscan_compare into one binary that dispatches over the matrix's full dimensionality spine
// {1,2,3,4,6,8,12,16,32,64,128} (design section 8.2). Build in a Release config; not a ctest.
//
// stdout:  RESULT eps=<f> ordering_ms=<i> n_clusters=<i> noise=<i>
// --out-labels file: header "label", then one int per input point (-1 = noise).
//
// CLI (see the contract documented in tools/run_matrix.py):
//   optics_matrix --coords <csv> --algo optics|soptics|hdbscan|shdbscan --out-labels <csv>
//     [--min-pts 16] [--min-cluster-size 16] [--chi 0.05] [--eps knee|uniform|<num>]
//     [--metric cosine|l2|l1|euclidean] [--mode ondemand|precompute]
//     [--backend exact|approx100|approx500|approx1000|hnsw]
//     [--projection gaussian|structured] [--threads 4] [--seed 42]
//   For optics, --eps/--mode/--backend apply (the D4/D3/D1 sweep axes). 'hnsw' needs the binary
//   built with -DOPTICS_ENABLE_HNSW=ON; eps is always computed with the exact backend so every
//   backend searches the same radius (D1 measures search performance + recall, not a new radius).
//   For soptics/shdbscan, --metric/--projection/--seed apply and
//   eps auto-scales (the #58 data-scaled default). For hdbscan/shdbscan, --min-cluster-size and
//   --min-pts (as min_samples; 0 => min_cluster_size) apply.

#include <optics/optics.hpp>
#include <optics/hdbscan.hpp>
#include <optics/io.hpp>
#include <optics/Stopwatch.hpp>
#ifdef OPTICS_ENABLE_HNSW
#include <optics/hnsw_backend.hpp>  // D1 backend axis: approximate HNSW (gated by the CMake option)
#endif

#include "bench_config.hpp"
#include "csv_points.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace sw = stopwatch;

namespace {

struct Args {
    std::map<std::string, std::string> kv;
    std::string get( const std::string& k, const std::string& def = "" ) const {
        auto it = kv.find( k );
        return it == kv.end() ? def : it->second;
    }
    long get_i( const std::string& k, long def ) const {
        auto it = kv.find( k );
        return it == kv.end() ? def : std::stol( it->second );
    }
};

optics::Metric parse_metric( const std::string& s ) {
    if ( s == "l1" ) { return optics::Metric::L1; }
    if ( s == "l2" || s == "euclidean" ) { return optics::Metric::L2; }
    return optics::Metric::Cosine;
}

void write_labels( const std::string& path, const std::vector<long long>& labels ) {
    std::ofstream out( path );
    out << "label\n";
    for ( const int l : labels ) { out << l << "\n"; }
}

template <typename LabelT>
void summarize( const std::vector<LabelT>& labels, int& n_clusters, int& noise ) {
    std::vector<LabelT> seen;
    noise = 0;
    for ( const auto l : labels ) {
        if ( l < 0 ) { ++noise; continue; }
        bool found = false;
        for ( const auto s : seen ) { if ( s == l ) { found = true; break; } }
        if ( !found ) { seen.push_back( l ); }
    }
    n_clusters = static_cast<int>( seen.size() );
}

template <std::size_t Dim>
int run( const Args& a, const std::vector<double>& flat, std::size_t n ) {
    const auto pts = bench::pack<Dim>( flat, n );
    const std::string algo = a.get( "--algo", "optics" );
    const std::size_t min_pts = static_cast<std::size_t>( a.get_i( "--min-pts", 16 ) );
    const std::size_t mcs = static_cast<std::size_t>( a.get_i( "--min-cluster-size", 16 ) );
    const double chi = std::stod( a.get( "--chi", "0.05" ) );
    const unsigned threads = static_cast<unsigned>( a.get_i( "--threads", bench::threads() ) );
    const unsigned seed = static_cast<unsigned>( a.get_i( "--seed", 42 ) );
    const std::string labels_out = a.get( "--out-labels", "" );

    double eps_used = -1.0;
    long long ordering_ms = 0;
    std::vector<long long> labels;

    if ( algo == "optics" || algo == "soptics" ) {
        std::vector<optics::reachability_dist> reach;
        if ( algo == "optics" ) {
            const std::string eps_spec = a.get( "--eps", "knee" );
            eps_used = ( eps_spec == "knee" )    ? optics::epsilon_estimation_knee<double, Dim>( pts, min_pts )
                       : ( eps_spec == "uniform" ) ? optics::epsilon_estimation( pts, min_pts )
                       : std::atof( eps_spec.c_str() );
            const auto mode = ( a.get( "--mode", "ondemand" ) == "precompute" )
                                  ? optics::NeighborMode::Precompute
                                  : optics::NeighborMode::OnDemand;
            // D1 backend axis. eps_used (computed above with the exact backend) is passed to every
            // backend, so the comparison is pure search performance + recall, not a different radius.
            const std::string be = a.get( "--backend", "exact" );
            sw::Stopwatch w;
            if ( be == "exact" ) {
                reach = optics::compute_reachability_dists<double, Dim>( pts, min_pts, eps_used, mode, threads );
            } else if ( be == "approx100" ) {
                reach = optics::compute_reachability_dists<double, Dim, optics::ApproxNanoflannBackend<double, Dim, 100>>(
                    pts, min_pts, eps_used, mode, threads );
            } else if ( be == "approx500" ) {
                reach = optics::compute_reachability_dists<double, Dim, optics::ApproxNanoflannBackend<double, Dim, 500>>(
                    pts, min_pts, eps_used, mode, threads );
            } else if ( be == "approx1000" ) {
                reach = optics::compute_reachability_dists<double, Dim, optics::ApproxNanoflannBackend<double, Dim, 1000>>(
                    pts, min_pts, eps_used, mode, threads );
            } else if ( be == "hnsw" ) {
#ifdef OPTICS_ENABLE_HNSW
                reach = optics::compute_reachability_dists<double, Dim, optics::HnswBackend<double, Dim>>(
                    pts, min_pts, eps_used, mode, threads );
#else
                std::cerr << "backend 'hnsw' not built (configure -DOPTICS_ENABLE_HNSW=ON)\n";
                return 3;
#endif
            } else {
                std::cerr << "unknown --backend " << be << " (exact|approx100|approx500|approx1000|hnsw)\n";
                return 2;
            }
            ordering_ms = static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
        } else {  // soptics: eps auto-scales (#58); metric / projection apply.
            const auto metric = parse_metric( a.get( "--metric", "cosine" ) );
            const auto proj = ( a.get( "--projection", "gaussian" ) == "structured" )
                                  ? optics::SopticsProjection::Structured
                                  : optics::SopticsProjection::Gaussian;
            sw::Stopwatch w;
            reach = optics::compute_soptics_reachability_dists<double, Dim>(
                pts, min_pts, -1.0, 1024u, 0u, std::size_t{ 0 }, seed, threads, metric, 0.0,
                std::vector<std::size_t>{}, proj );
            ordering_ms = static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
        }
        labels = optics::io::cluster_labels(
            n, optics::get_cluster_indices( reach, optics::get_chi_clusters_flat( reach, chi, min_pts, 0.0, mcs ) ) );
    } else if ( algo == "hdbscan" || algo == "shdbscan" ) {
        const std::size_t ms = ( min_pts == 0 ) ? mcs : min_pts;
        if ( algo == "hdbscan" ) {
            sw::Stopwatch w;
            const auto r = optics::hdbscan<double, Dim>( pts, mcs, ms );
            ordering_ms = static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
            labels.assign( r.labels.begin(), r.labels.end() );
        } else {
            const auto metric = parse_metric( a.get( "--metric", "cosine" ) );
            const auto proj = ( a.get( "--projection", "gaussian" ) == "structured" )
                                  ? optics::SopticsProjection::Structured
                                  : optics::SopticsProjection::Gaussian;
            sw::Stopwatch w;
            const auto r = optics::shdbscan<double, Dim>(
                pts, mcs, ms, -1.0, 1024u, 0u, std::size_t{ 0 }, seed, threads,
                optics::ClusterSelectionMethod::EOM, false, metric, 0.0, proj );
            ordering_ms = static_cast<long long>( bench::ceil_ms_from_us( w.elapsed<sw::mus>() ) );
            labels.assign( r.labels.begin(), r.labels.end() );
        }
    } else {
        std::cerr << "unknown --algo " << algo << " (optics|soptics|hdbscan|shdbscan)\n";
        return 2;
    }

    int n_clusters = 0, noise = 0;
    summarize( labels, n_clusters, noise );
    if ( !labels_out.empty() ) { write_labels( labels_out, labels ); }
    std::cout << "RESULT eps=" << eps_used << " ordering_ms=" << ordering_ms
              << " n_clusters=" << n_clusters << " noise=" << noise << "\n";
    return 0;
}

}  // namespace

int main( int argc, char** argv ) {
    Args a;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        if ( std::string( argv[i] ).rfind( "--", 0 ) == 0 ) { a.kv[argv[i]] = argv[i + 1]; }
    }
    const std::string coords = a.get( "--coords", "" );
    if ( coords.empty() ) {
        std::cerr << "usage: optics_matrix --coords <csv> --algo optics|soptics|hdbscan|shdbscan "
                     "--out-labels <csv> [--min-pts 16] [--min-cluster-size 16] [--eps knee] ...\n";
        return 2;
    }
    std::vector<double> flat;
    std::size_t n = 0, dim = 0;
    if ( !bench::read_csv( coords, flat, n, dim ) ) { std::cerr << "cannot read " << coords << "\n"; return 2; }
    switch ( dim ) {
        case 1:   return run<1>( a, flat, n );
        case 2:   return run<2>( a, flat, n );
        case 3:   return run<3>( a, flat, n );
        case 4:   return run<4>( a, flat, n );
        case 6:   return run<6>( a, flat, n );
        case 8:   return run<8>( a, flat, n );
        case 12:  return run<12>( a, flat, n );
        case 16:  return run<16>( a, flat, n );
        case 32:  return run<32>( a, flat, n );
        case 64:  return run<64>( a, flat, n );
        case 128: return run<128>( a, flat, n );
        default:
            std::cerr << "unsupported dim " << dim << " (supported: 1,2,3,4,6,8,12,16,32,64,128)\n";
            return 2;
    }
}
