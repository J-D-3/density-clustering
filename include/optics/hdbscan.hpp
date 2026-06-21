// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// HDBSCAN* -- Hierarchical Density-Based Spatial Clustering of Applications with Noise
// (Campello, Moulavi & Sander 2013; accelerated by McInnes & Healy 2017). GitHub issue #52.
//
// Where OPTICS (optics.hpp) produces a 1-D cluster-ordering and asks you to pick an epsilon
// threshold or run the Xi steep-area extractor, HDBSCAN* instead builds a cluster *hierarchy*
// and condenses it into a small set of the most persistent ("stable") clusters -- so you only
// supply `min_cluster_size` (and an optional density smoother `min_samples`); no eps, no
// reachability cut. It naturally finds clusters of differing densities and labels the rest noise.
//
// The pipeline (see the named detail:: helpers below):
//   1. core distance      core_k(x) = distance to x's k-th nearest neighbour (k = min_samples),
//                         a local density estimate. Reuses the backend's knn_core_dist -- the
//                         very same quantity OPTICS already computes.
//   2. mutual reachability d_mreach(a,b) = max(core_k(a), core_k(b), d(a,b)). SYMMETRIC, unlike
//                         OPTICS's directed reachability max(core(o), d(o,p)). It "lifts the sea
//                         level" of noise so single-linkage stops chaining through sparse gaps.
//   3. MST                a minimum spanning tree of the complete mutual-reachability graph
//                         (dense Prim -- the density-connectivity backbone of the data).
//   4. hierarchy          sort MST edges ascending + union-find => a single-linkage dendrogram.
//   5. condense           walk the dendrogram top-down; at each split, a side smaller than
//                         min_cluster_size just "falls out" as noise rather than being a real
//                         split. Yields a compact condensed tree annotated with lambda = 1/dist.
//   6. extract            select the most stable clusters by Excess of Mass (default) or take the
//                         leaves, then label points (-1 == noise) with a membership probability.
//
// Like the rest of the library this is header-only, dependency-free (only the vendored backend),
// and templated over the coordinate type (float/double) and dimension. The condensed-tree /
// stability machinery (steps 4-6) is deliberately METRIC- and MST-agnostic, so a future faster
// or approximate MST (dual-tree Boruvka, or the sOPTICS/CEOs graph) can feed the same extractor.
//
// Reference implementations consulted for the algorithm (not copied -- this is reimplemented from
// the papers to stay MIT-clean): scikit-learn-contrib/hdbscan, TutteInstitute/fast_hdbscan,
// sklearn.cluster.HDBSCAN. CONVENTION NOTE: `min_samples` here is self-inclusive (the point
// counts as its own 1st neighbour), matching sklearn.cluster.HDBSCAN and this library's OPTICS
// core-distance; scikit-learn-contrib/hdbscan excludes self, so add 1 there to match.

#include "backend.hpp"
#include "preprocess.hpp"               // deduplicate (auto-dedup / weighted path, issue #46)
#include "detail/math.hpp"
#include "detail/thread_pool.hpp"
#include "detail/random_features.hpp"   // Metric / SopticsProjection enums + kernel embedding (sHDBSCAN)
#include "detail/random_projection.hpp" // CEOs approximate neighbor graph (sHDBSCAN)
#include "detail/boruvka_mst.hpp"       // exact sub-quadratic Boruvka MST (issue #66, Phase 2)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace optics {

// How the flat clusters are picked from the condensed tree.
//   EOM  : Excess of Mass -- the standard HDBSCAN* rule; keeps the most persistent clusters,
//          which may sit at different depths of the hierarchy.
//   Leaf : take the leaf clusters of the condensed tree -- the most fine-grained, homogeneous
//          partition (more, smaller clusters).
enum class ClusterSelectionMethod { EOM, Leaf };

// How the exact mutual-reachability MST is built (issue #66). Both feed the identical
// MST-agnostic extraction tail; they differ only in how the spanning tree is produced.
//   DensePrim : the textbook O(n^2) dense Prim over the COMPLETE graph -- exact, simple, the right
//               tool up to n ~ 1e4 (the default; preserves the pinned hdbscan: results bit-for-bit).
//   KnnGraph  : near-exact, sub-quadratic. Builds each point's exact k-NN graph (one query also
//               yields the core distance), forms mutual-reachability edges over that sparse graph,
//               and runs Kruskal with a connectivity fix-up -- the exact-Euclidean cousin of the
//               CEOs graph sHDBSCAN uses. Core distances are exact; only the long inter-cluster
//               edges can differ from the true MST (they fall at the top of the hierarchy, where
//               well-separated clusters split anyway), so validate by ARI vs DensePrim. Requires a
//               backend modeling KnnGraph (NanoflannBackend does); falls back to DensePrim otherwise.
//   Boruvka   : EXACT and sub-quadratic -- the same minimum spanning tree as DensePrim (identical
//               total weight), built by Boruvka's algorithm over a purpose-built component-aware
//               KD-tree (see detail/boruvka_mst.hpp). This is the one to use when you need both
//               exactness AND scale (n past the ~1e4 dense-Prim ceiling). Works with any KnnCoreDist
//               backend (the tree is internal; the backend only supplies core distances).
enum class MstAlgorithm { DensePrim, KnnGraph, Boruvka };

// Result of a HDBSCAN* run. `labels` and `probabilities` are parallel to the input points.
struct HdbscanResult {
	std::vector<int> labels;            // per point: cluster id 0..n_clusters-1, or -1 for noise
	std::vector<double> probabilities;  // per point: membership strength in [0,1] (0 for noise)
	std::size_t n_clusters = 0;
};


namespace detail {

// One undirected edge of the mutual-reachability MST.
struct MstEdge {
	std::size_t u = 0;
	std::size_t v = 0;
	double weight = 0.0;  // d_mreach(u,v)
};

// One merge of the single-linkage dendrogram (step 4). Node ids 0..n-1 are the original points;
// ids n..2n-2 are merges. Row i (for merge node n+i) records the two child node ids it joins, the
// mutual-reachability distance at which they joined, and the total point count of the merged node.
struct LinkageRow {
	std::size_t left = 0;
	std::size_t right = 0;
	double dist = 0.0;
	std::size_t size = 0;
};

// One edge of the condensed tree (step 5): `child` left/fell out of cluster `parent` at density
// level lambda_val ( = 1/dist ), carrying child_size points. `parent` is always a cluster id
// (>= n_points); `child` is either a sub-cluster id (>= n_points) or an original point (< n_points).
struct CondensedEdge {
	std::size_t parent = 0;
	std::size_t child = 0;
	double lambda_val = 0.0;
	std::size_t child_size = 0;
};


//=== Step 1: core distances =================================================

// core_k(x) for every point: the distance to its min_samples-th nearest neighbour (self included),
// computed in parallel via the backend's k-NN. A point with fewer than min_samples points in the
// whole cloud gets core distance 0 (it cannot be a denser-than-noise centre anyway).
//
// Weighted (issue #46): when `weights` is non-empty each point stands for `weights[i]` identical
// originals, and the core distance becomes the distance at which the cumulative neighbour weight
// (nearest first, self included) reaches min_samples -- matching scikit-learn's `sample_weight`
// (and reducing to the unweighted k-th distance when every weight is 1). Needs a backend modeling
// KnnCoreDistWeighted; without it (and without weights) the plain k-NN path is used.
template <class T, std::size_t Dim, class Backend>
std::vector<double> hdbscan_core_distances( const std::vector<std::array<T, Dim>>& points,
											 std::size_t min_samples, unsigned n_threads,
											 const std::vector<std::size_t>& weights = {} ) {
	static_assert( KnnCoreDist<Backend, T, Dim>,
		"hdbscan requires a backend modeling KnnCoreDist (e.g. NanoflannBackend)" );
	const std::size_t n = points.size();
	std::vector<double> core( n, 0.0 );
	const Backend backend( points );
	const T no_cap = std::numeric_limits<T>::max();  // no radius cap: a plain k-NN distance
	const bool weighted = !weights.empty();
	detail::parallel_for( n_threads, n, [&]( std::size_t i ) {
		if ( weighted ) {
			if constexpr ( KnnCoreDistWeighted<Backend, T, Dim> ) {
				const auto cd = backend.knn_core_dist_weighted( points[i], weights, min_samples );
				core[i] = cd.has_value() ? *cd : 0.0;
				return;
			}
		}
		const auto cd = backend.knn_core_dist( points[i], min_samples, no_cap );
		core[i] = cd.has_value() ? *cd : 0.0;
	} );
	return core;
}


//=== Step 3: mutual-reachability MST (dense Prim) ===========================

// Exact MST of the COMPLETE mutual-reachability graph. We use the textbook dense (array) Prim:
// O(n^2) time, O(n) memory. For a *complete* graph this is the right tool -- a binary-heap Prim
// would only add a log factor and heavy churn (every vertex is always reachable, so there is no
// sparsity to exploit). It reuses the library's existing pieces: detail::dist for the metric and
// the precomputed core[] for the density lift. Returns the n-1 MST edges (unsorted).
//
// Tie-break: among equal candidate weights the lowest-index vertex is chosen (the `<` below is
// strict + first-found), so the MST -- and thus the whole clustering -- is deterministic.
template <class T, std::size_t Dim>
std::vector<MstEdge> mutual_reachability_mst( const std::vector<std::array<T, Dim>>& points,
											  const std::vector<double>& core ) {
	const std::size_t n = points.size();
	std::vector<MstEdge> mst;
	if ( n < 2 ) { return mst; }
	mst.reserve( n - 1 );

	constexpr double inf = std::numeric_limits<double>::infinity();
	std::vector<char> in_tree( n, 0 );
	std::vector<double> best( n, inf );       // cheapest mutual-reach edge joining v to the tree
	std::vector<std::size_t> parent( n, 0 );  // the tree endpoint of that edge
	best[0] = 0.0;                             // grow the tree from point 0

	for ( std::size_t iter = 0; iter < n; ++iter ) {
		// Pick the not-yet-in-tree vertex closest to the tree.
		std::size_t u = n;
		double u_best = inf;
		for ( std::size_t v = 0; v < n; ++v ) {
			if ( !in_tree[v] && best[v] < u_best ) { u_best = best[v]; u = v; }
		}
		// u == n cannot happen on a complete graph (every vertex stays reachable), but guard anyway.
		if ( u == n ) { break; }
		in_tree[u] = 1;
		if ( iter != 0 ) { mst.push_back( { parent[u], u, best[u] } ); }

		// Relax: update every outside vertex's cheapest connection through the newly added u.
		const double core_u = core[u];
		for ( std::size_t v = 0; v < n; ++v ) {
			if ( in_tree[v] ) { continue; }
			const double d = detail::dist( points[u], points[v] );
			const double mreach = std::max( std::max( core_u, core[v] ), d );
			if ( mreach < best[v] ) { best[v] = mreach; parent[v] = u; }
		}
	}
	return mst;
}


//=== Steps 1+3, approximate (sHDBSCAN): core distance + MST from a sparse graph ===

// k-th smallest of a list of squared distances (self at 0 included), as a real distance; nullopt
// if fewer than k entries. A small standalone copy of optics.hpp's compute_core_dist_from_sq so
// hdbscan.hpp need not pull in the whole OPTICS header for the approximate path.
inline std::optional<double> kth_dist_from_sq( const std::vector<double>& sq, std::size_t k ) {
	if ( sq.size() < k || k == 0 ) { return std::nullopt; }
	std::vector<double> scratch( sq );
	std::nth_element( scratch.begin(), scratch.begin() + ( k - 1 ), scratch.end() );
	return std::sqrt( scratch[k - 1] );
}

// Weighted k-th distance: sort the (squared distance, weight) candidate pairs by distance and
// return the distance at which the cumulative weight first reaches min_samples; nullopt if the
// total weight stays below it. Reduces to kth_dist_from_sq when every weight is 1.
inline std::optional<double> kth_weighted_dist_from_sq( const std::vector<double>& sq,
														const std::vector<std::size_t>& neighbor_indices,
														const std::vector<std::size_t>& weights, std::size_t min_samples ) {
	std::vector<std::pair<double, std::size_t>> dw;
	dw.reserve( sq.size() );
	for ( std::size_t j = 0; j < sq.size(); ++j ) { dw.emplace_back( sq[j], weights[neighbor_indices[j]] ); }
	std::sort( dw.begin(), dw.end(), []( const auto& a, const auto& b ) { return a.first < b.first; } );
	std::size_t acc = 0;
	for ( const auto& [s, w] : dw ) { acc += w; if ( acc >= min_samples ) { return std::sqrt( s ); } }
	return std::nullopt;
}

// Approximate core distances from the CEOs candidate squared distances (parallel to the neighbor
// lists). core_k(i) = the min_samples-th nearest among i's candidates. If a point has fewer than
// min_samples candidates (the approximation found too few), fall back to its farthest candidate --
// a conservative over-estimate that makes such a point connect late (and likely become noise).
//
// Weighted (issue #46): when `weights` is non-empty the k-th becomes the distance at which the
// cumulative candidate weight reaches min_samples (`neighbors` supplies each candidate's point index
// to look up its weight). Empty weights => the plain count-based k-th, unchanged.
inline std::vector<double> approx_core_distances( const std::vector<std::vector<double>>& neighbor_sq,
												  std::size_t min_samples,
												  const std::vector<std::vector<std::size_t>>& neighbors = {},
												  const std::vector<std::size_t>& weights = {} ) {
	const std::size_t n = neighbor_sq.size();
	const bool weighted = !weights.empty();
	std::vector<double> core( n, 0.0 );
	for ( std::size_t i = 0; i < n; ++i ) {
		const auto cd = weighted ? kth_weighted_dist_from_sq( neighbor_sq[i], neighbors[i], weights, min_samples )
								 : kth_dist_from_sq( neighbor_sq[i], min_samples );
		if ( cd.has_value() ) {
			core[i] = *cd;
		} else {
			double mx = 0.0;
			for ( const double s : neighbor_sq[i] ) { mx = std::max( mx, s ); }
			core[i] = std::sqrt( mx );
		}
	}
	return core;
}

// Approximate mutual-reachability MST built from the sparse, symmetric CEOs neighbor graph instead
// of the complete graph (sHDBSCAN). Each candidate pair (i,j) contributes a mutual-reachability
// edge max(core(i),core(j),d(i,j)); Kruskal over those sparse edges builds the MST. The CEOs graph
// can be disconnected (it only ever lists a point's approximate neighbours), so any leftover
// components are joined to point 0's tree with a weight strictly larger than every real edge --
// those merges then sit at the very top of the hierarchy, exactly where well-separated clusters (and
// tiny stray components, which fall below min_cluster_size and become noise) ought to split.
//
// `neighbors`/`neighbor_sq` are the symmetric CEOs lists (each includes the point itself); because
// the relation is symmetric every undirected edge is seen from its lower-index endpoint, so taking
// only j > i lists each edge exactly once.
// Kruskal MST over a precomputed list of candidate mutual-reachability edges, with a connectivity
// fix-up: any components the candidate graph leaves disconnected are joined to point 0's tree with a
// weight strictly larger than every real edge, so those merges sit at the very top of the hierarchy
// (exactly where well-separated clusters -- and tiny stray components that fall below
// min_cluster_size and become noise -- ought to split). Shared by both sparse-graph MST builders:
// the CEOs graph (sHDBSCAN, approximate neighbors) and the exact k-NN graph (issue #66). `edges` is
// consumed (sorted in place); the caller owns building the edge list for its graph.
inline std::vector<MstEdge> sparse_graph_mst( std::vector<MstEdge>& edges, std::size_t n ) {
	std::vector<MstEdge> mst;
	if ( n < 2 ) { return mst; }
	std::sort( edges.begin(), edges.end(), []( const MstEdge& a, const MstEdge& b ) { return a.weight < b.weight; } );

	// Kruskal with a plain union-find over the n points.
	std::vector<std::size_t> uf( n );
	std::iota( uf.begin(), uf.end(), std::size_t( 0 ) );
	auto find = [&]( std::size_t x ) {
		while ( uf[x] != x ) { uf[x] = uf[uf[x]]; x = uf[x]; }  // path halving
		return x;
	};

	mst.reserve( n - 1 );
	double max_w = 0.0;
	for ( const auto& e : edges ) {
		const std::size_t ra = find( e.u );
		const std::size_t rb = find( e.v );
		if ( ra != rb ) { uf[ra] = rb; mst.push_back( e ); max_w = std::max( max_w, e.weight ); }
	}

	// Connect any remaining components to point 0's tree above all real edges.
	if ( mst.size() + 1 < n ) {
		const double connect_w = ( max_w > 0.0 ? 2.0 * max_w : 1.0 ) + 1.0;
		const std::size_t anchor = find( 0 );
		for ( std::size_t i = 1; i < n; ++i ) {
			if ( find( i ) != find( anchor ) ) {
				uf[find( i )] = find( anchor );
				mst.push_back( { std::size_t( 0 ), i, connect_w } );
			}
		}
	}
	return mst;
}

inline std::vector<MstEdge> approx_mutual_reachability_mst(
		const std::vector<std::vector<std::size_t>>& neighbors,
		const std::vector<std::vector<double>>& neighbor_sq, const std::vector<double>& core,
		std::size_t n ) {
	if ( n < 2 ) { return {}; }
	// Gather candidate mutual-reachability edges (each undirected pair once, from the lower index --
	// the CEOs relation is symmetric, so every edge appears on its lower endpoint's list).
	std::vector<MstEdge> edges;
	for ( std::size_t i = 0; i < n; ++i ) {
		const auto& nb = neighbors[i];
		const auto& sq = neighbor_sq[i];
		for ( std::size_t t = 0; t < nb.size(); ++t ) {
			const std::size_t j = nb[t];
			if ( j <= i ) { continue; }  // skip self and the mirror copy on j's list
			const double d = std::sqrt( sq[t] );
			const double w = std::max( std::max( core[i], core[j] ), d );
			edges.push_back( { i, j, w } );
		}
	}
	return sparse_graph_mst( edges, n );
}


//=== Steps 1+3, exact-but-sub-quadratic (issue #66): MST from an exact k-NN graph ===

// Near-exact mutual-reachability MST built from each point's EXACT k-NN graph instead of the complete
// graph (the DensePrim alternative) -- sub-quadratic, for the n > ~1e4 regime where O(n^2) Prim is
// too slow. One k-NN query per point (parallel via the backend) yields both the exact core distance
// (the min_samples-th neighbor) and that point's candidate MST edges, mirroring fast_hdbscan's
// "initialise Boruvka from the kNN". Mutual-reachability edges are formed over the kNN graph and fed
// to the shared sparse_graph_mst (Kruskal + connectivity fix-up). The k-NN graph is directed (i may
// be in j's list without the reverse), so every candidate edge is listed in BOTH directions and the
// union-find dedups -- no symmetrization pass needed.
//
// Exactness: core distances are EXACT (graph_k >= min_samples). The tree is exact whenever every true
// MST edge joins points that are mutual k-NNs -- true within dense cluster cores; the few long
// inter-cluster edges that are not captured are supplied by the connectivity fix-up above all real
// edges, so the cluster *partition* is preserved even when the raw tree differs. Validate by ARI vs
// DensePrim, not bit-identity. graph_k = 0 => an auto default (~2*min_samples, floor 10).
//
// `weights` (issue #46): non-empty makes the core distance weight-aware (cumulative neighbor weight
// reaches min_samples), reusing approx_core_distances' weighted path. Requires KnnGraph<Backend>.
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
std::vector<MstEdge> knn_graph_mst( const std::vector<std::array<T, Dim>>& points, std::size_t min_samples,
									unsigned n_threads, const std::vector<std::size_t>& weights = {},
									std::size_t graph_k = 0 ) {
	static_assert( KnnGraph<Backend, T, Dim>,
		"knn_graph_mst requires a backend modeling KnnGraph (e.g. NanoflannBackend)" );
	const std::size_t n = points.size();
	if ( n < 2 ) { return {}; }

	// A point's MST neighbour usually lies among its few nearest, but mutual reachability can pull it
	// a little further, so query a touch beyond min_samples (capped at n). Larger graph_k -> closer to
	// the true MST at higher query cost; the connectivity fix-up covers whatever the graph still misses.
	const std::size_t k = ( graph_k > 0 )
		? std::min<std::size_t>( n, graph_k )
		: std::min<std::size_t>( n, std::max<std::size_t>( 2 * min_samples, 10 ) );

	const Backend backend( points );
	std::vector<std::vector<std::size_t>> neighbors( n );
	std::vector<std::vector<double>> neighbor_sq( n );
	detail::parallel_for( n_threads, n, [&]( std::size_t i ) {
		backend.knn_graph( points[i], k, neighbors[i], neighbor_sq[i] );
	} );

	// Exact core distances from the k-NN squared distances (reuses the sHDBSCAN helper: the
	// min_samples-th nearest, weight-aware when weights is non-empty).
	const auto core = weights.empty()
		? approx_core_distances( neighbor_sq, min_samples )
		: approx_core_distances( neighbor_sq, min_samples, neighbors, weights );

	// Mutual-reachability edges over the (directed) k-NN graph; list each direction and let Kruskal
	// dedup. Skip only the self-edge (the point's own 0-distance entry).
	std::vector<MstEdge> edges;
	edges.reserve( n * k );
	for ( std::size_t i = 0; i < n; ++i ) {
		const auto& nb = neighbors[i];
		const auto& sq = neighbor_sq[i];
		for ( std::size_t t = 0; t < nb.size(); ++t ) {
			const std::size_t j = nb[t];
			if ( j == i ) { continue; }
			const double d = std::sqrt( sq[t] );
			const double w = std::max( std::max( core[i], core[j] ), d );
			edges.push_back( { i, j, w } );
		}
	}
	return sparse_graph_mst( edges, n );
}


//=== Step 4: single-linkage hierarchy from the MST ==========================

// Union-find that mints a fresh node id on every merge (the dendrogram's internal nodes). find()
// returns the id of the current top-most node containing x; do_union() joins two such top nodes
// under a brand-new id. This is the standard HDBSCAN single-linkage labelling.
struct LinkageUnionFind {
	std::vector<std::size_t> parent;  // size 2N-1; leaves 0..N-1, internal nodes N..2N-2
	std::vector<std::size_t> size;
	std::size_t next_label;

	// `weights` (optional): each leaf 0..N-1 starts with its point weight instead of 1, so every
	// node's `size` is the total ORIGINAL-point weight under it (issue #46). Empty => all-ones.
	explicit LinkageUnionFind( std::size_t N, const std::vector<std::size_t>& weights = {} )
			: parent( 2 * N - 1 ), size( 2 * N - 1, 1 ), next_label( N ) {
		for ( std::size_t i = 0; i < parent.size(); ++i ) { parent[i] = i; }
		if ( !weights.empty() ) {
			for ( std::size_t i = 0; i < N; ++i ) { size[i] = weights[i]; }
		}
	}
	std::size_t find( std::size_t x ) {
		std::size_t root = x;
		while ( parent[root] != root ) { root = parent[root]; }
		while ( parent[x] != root ) { const std::size_t nx = parent[x]; parent[x] = root; x = nx; }  // path compression
		return root;
	}
	void do_union( std::size_t a, std::size_t b ) {  // a, b must be current top node ids
		parent[a] = next_label;
		parent[b] = next_label;
		size[next_label] = size[a] + size[b];
		++next_label;
	}
};

// Sort the MST edges ascending and replay them through the union-find, producing the N-1 merge
// rows of the single-linkage dendrogram. The final merge (node 2N-2) is the root. `weights`
// (optional) makes each row's size a total point WEIGHT rather than a count (issue #46).
inline std::vector<LinkageRow> mst_to_linkage( std::vector<MstEdge> edges, std::size_t N,
											   const std::vector<std::size_t>& weights = {} ) {
	std::vector<LinkageRow> rows;
	if ( N < 2 ) { return rows; }
	std::sort( edges.begin(), edges.end(), []( const MstEdge& a, const MstEdge& b ) { return a.weight < b.weight; } );
	LinkageUnionFind uf( N, weights );
	rows.reserve( edges.size() );
	for ( const auto& e : edges ) {
		const std::size_t ra = uf.find( e.u );
		const std::size_t rb = uf.find( e.v );
		rows.push_back( { ra, rb, e.weight, uf.size[ra] + uf.size[rb] } );
		uf.do_union( ra, rb );
	}
	return rows;
}


//=== Step 5: condense the cluster tree ======================================

// Total point WEIGHT under a linkage node: a leaf (id < N) weighs weights[node] (1 when unweighted),
// an internal node carries the merge's accumulated size.
inline std::size_t linkage_node_size( const std::vector<LinkageRow>& rows, std::size_t N, std::size_t node,
									  const std::vector<std::size_t>& weights = {} ) {
	if ( node < N ) { return weights.empty() ? std::size_t( 1 ) : weights[node]; }
	return rows[node - N].size;
}

// Breadth-first list of all linkage-tree node ids in the subtree rooted at `root`.
inline std::vector<std::size_t> bfs_linkage( const std::vector<LinkageRow>& rows, std::size_t N, std::size_t root ) {
	std::vector<std::size_t> order;
	std::vector<std::size_t> frontier{ root };
	while ( !frontier.empty() ) {
		std::vector<std::size_t> next;
		for ( const std::size_t node : frontier ) {
			order.push_back( node );
			if ( node >= N ) {
				next.push_back( rows[node - N].left );
				next.push_back( rows[node - N].right );
			}
		}
		frontier.swap( next );
	}
	return order;
}

// Condense the single-linkage dendrogram into a small tree of genuine clusters (step 5). Walking
// top-down from the root, at every split we ask whether each side still has min_cluster_size
// points: a side that does not simply "falls out" (its points leave the current cluster as noise
// at this density level) rather than spawning a real cluster. New cluster ids start at N (the root
// is N) and increase as we descend. lambda_val = 1/dist is the density level of the event.
inline std::vector<CondensedEdge> condense_tree( const std::vector<LinkageRow>& rows, std::size_t N,
												 std::size_t min_cluster_size,
												 const std::vector<std::size_t>& weights = {} ) {
	std::vector<CondensedEdge> result;
	if ( N == 0 ) { return result; }
	if ( N == 1 ) { return result; }  // a single point: no hierarchy, handled as noise upstream

	const std::size_t root = 2 * N - 2;
	std::vector<std::size_t> relabel( root + 1, 0 );
	relabel[root] = N;
	std::size_t next_label = N + 1;
	std::vector<char> ignore( root + 1, 0 );

	const auto lambda_of = []( double dist ) -> double {
		return dist > 0.0 ? 1.0 / dist : std::numeric_limits<double>::infinity();
	};
	// Weight a single point falls out with: its sample weight (issue #46), or 1 unweighted.
	const auto point_weight = [&]( std::size_t p ) -> std::size_t { return weights.empty() ? std::size_t( 1 ) : weights[p]; };

	for ( const std::size_t node : bfs_linkage( rows, N, root ) ) {
		if ( ignore[node] || node < N ) { continue; }
		const LinkageRow& row = rows[node - N];
		const std::size_t left = row.left;
		const std::size_t right = row.right;
		const double lambda_value = lambda_of( row.dist );
		const std::size_t left_count = linkage_node_size( rows, N, left, weights );
		const std::size_t right_count = linkage_node_size( rows, N, right, weights );

		if ( left_count >= min_cluster_size && right_count >= min_cluster_size ) {
			// Genuine split: both sides become new clusters.
			relabel[left] = next_label++;
			result.push_back( { relabel[node], relabel[left], lambda_value, left_count } );
			relabel[right] = next_label++;
			result.push_back( { relabel[node], relabel[right], lambda_value, right_count } );
		} else if ( left_count < min_cluster_size && right_count < min_cluster_size ) {
			// Both sides too small: every point falls out of the current cluster here.
			for ( const std::size_t sub : bfs_linkage( rows, N, left ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, point_weight( sub ) } ); }
				ignore[sub] = 1;
			}
			for ( const std::size_t sub : bfs_linkage( rows, N, right ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, point_weight( sub ) } ); }
				ignore[sub] = 1;
			}
		} else if ( left_count < min_cluster_size ) {
			// Left too small falls out; right continues as the SAME cluster.
			relabel[right] = relabel[node];
			for ( const std::size_t sub : bfs_linkage( rows, N, left ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, point_weight( sub ) } ); }
				ignore[sub] = 1;
			}
		} else {
			// Right too small falls out; left continues as the SAME cluster.
			relabel[left] = relabel[node];
			for ( const std::size_t sub : bfs_linkage( rows, N, right ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, point_weight( sub ) } ); }
				ignore[sub] = 1;
			}
		}
	}
	return result;
}


//=== Step 6a: cluster stabilities ===========================================

// Stability S(C) = sum over points p of C of ( lambda_p - lambda_birth(C) ), accumulated edge by
// edge: every child of C (point or sub-cluster) leaves C at its edge's lambda carrying child_size
// points. lambda_birth(C) is the lambda at which C itself appeared as a child of its parent (the
// root's birth is 0). Returned as cluster-id -> stability.
inline std::unordered_map<std::size_t, double> compute_stabilities(
		const std::vector<CondensedEdge>& tree, std::size_t N ) {
	std::unordered_map<std::size_t, double> stability;
	if ( tree.empty() ) { return stability; }

	// births[c] = lambda at which cluster c was born (= the lambda of the edge where c is the child).
	std::size_t max_id = N;
	for ( const auto& e : tree ) { max_id = std::max( max_id, std::max( e.parent, e.child ) ); }
	std::vector<double> births( max_id + 1, 0.0 );
	for ( const auto& e : tree ) { births[e.child] = e.lambda_val; }
	births[N] = 0.0;  // the root cluster has no parent edge: birth at lambda 0

	for ( const auto& e : tree ) {
		stability[e.parent] += ( e.lambda_val - births[e.parent] ) * static_cast<double>( e.child_size );
	}
	return stability;
}


//=== Step 6b: cluster selection (Excess of Mass / leaf) =====================

// Sub-cluster children of each cluster (only edges whose child is itself a cluster, child >= N).
inline std::unordered_map<std::size_t, std::vector<std::size_t>> cluster_children(
		const std::vector<CondensedEdge>& tree, std::size_t N ) {
	std::unordered_map<std::size_t, std::vector<std::size_t>> children;
	for ( const auto& e : tree ) {
		if ( e.child >= N ) { children[e.parent].push_back( e.child ); }
	}
	return children;
}

// Breadth-first list of cluster ids in the condensed subtree rooted at `start` (clusters only).
inline std::vector<std::size_t> bfs_clusters(
		const std::unordered_map<std::size_t, std::vector<std::size_t>>& children, std::size_t start ) {
	std::vector<std::size_t> order;
	std::vector<std::size_t> frontier{ start };
	while ( !frontier.empty() ) {
		std::vector<std::size_t> next;
		for ( const std::size_t c : frontier ) {
			order.push_back( c );
			const auto it = children.find( c );
			if ( it != children.end() ) { next.insert( next.end(), it->second.begin(), it->second.end() ); }
		}
		frontier.swap( next );
	}
	return order;
}

// Excess-of-Mass selection. Process clusters from the deepest (largest id) upward: if a cluster's
// children are collectively more stable than it, keep the children (and roll their stability up);
// otherwise select the cluster and deselect its whole subtree. The root (id N) is excluded unless
// allow_single_cluster, so the trivial "everything is one cluster" answer is not returned by default.
inline std::vector<std::size_t> select_eom(
		const std::vector<CondensedEdge>& tree, std::unordered_map<std::size_t, double> stability,
		std::size_t N, bool allow_single_cluster ) {
	// All cluster ids = the parents that appear in the tree, ascending.
	std::vector<std::size_t> clusters;
	for ( const auto& kv : stability ) { clusters.push_back( kv.first ); }
	std::sort( clusters.begin(), clusters.end() );
	if ( clusters.empty() ) { return {}; }

	const auto children = cluster_children( tree, N );
	std::unordered_map<std::size_t, bool> is_cluster;
	for ( const std::size_t c : clusters ) { is_cluster[c] = true; }

	// Descending order; exclude the root unless single clusters are allowed.
	for ( auto it = clusters.rbegin(); it != clusters.rend(); ++it ) {
		const std::size_t node = *it;
		if ( node == N && !allow_single_cluster ) { is_cluster[node] = false; continue; }
		double child_stability_sum = 0.0;
		const auto ch = children.find( node );
		if ( ch != children.end() ) {
			for ( const std::size_t c : ch->second ) { child_stability_sum += stability[c]; }
		}
		if ( child_stability_sum > stability[node] ) {
			is_cluster[node] = false;
			stability[node] = child_stability_sum;
		} else {
			for ( const std::size_t sub : bfs_clusters( children, node ) ) {
				if ( sub != node ) { is_cluster[sub] = false; }
			}
		}
	}

	std::vector<std::size_t> selected;
	for ( const std::size_t c : clusters ) { if ( is_cluster[c] ) { selected.push_back( c ); } }
	return selected;
}

// Leaf selection: every cluster that has no sub-cluster children (the most fine-grained partition).
// The root is included only if it is itself a leaf (no real split happened) and single clusters are
// allowed.
inline std::vector<std::size_t> select_leaf(
		const std::vector<CondensedEdge>& tree, const std::unordered_map<std::size_t, double>& stability,
		std::size_t N, bool allow_single_cluster ) {
	const auto children = cluster_children( tree, N );
	std::vector<std::size_t> clusters;
	for ( const auto& kv : stability ) { clusters.push_back( kv.first ); }
	std::sort( clusters.begin(), clusters.end() );

	std::vector<std::size_t> selected;
	for ( const std::size_t c : clusters ) {
		const auto it = children.find( c );
		const bool is_leaf = ( it == children.end() || it->second.empty() );
		if ( !is_leaf ) { continue; }
		if ( c == N && !allow_single_cluster ) { continue; }
		selected.push_back( c );
	}
	return selected;
}


//=== Step 6c: labelling + membership probabilities ==========================

// Turn the selected cluster ids into per-point labels and membership strengths. A point is mapped
// to the nearest selected cluster on the path from its immediate (fall-out) cluster up to the root;
// if none on that path is selected it is noise (-1). Membership = lambda_p / max(lambda) over the
// points sharing the point's final cluster -- 1.0 for the densest (last-to-leave) points, smaller
// for those that left earlier; exactly 1.0 (and finite) when the cluster's max lambda is 0 or inf.
inline void do_labelling( const std::vector<CondensedEdge>& tree, const std::vector<std::size_t>& selected,
						  std::size_t N, std::size_t n_points, std::vector<int>& labels,
						  std::vector<double>& probabilities ) {
	labels.assign( n_points, -1 );
	probabilities.assign( n_points, 0.0 );
	if ( selected.empty() ) { return; }

	// Selected cluster id -> compact label 0..k-1 (clusters already come in ascending id order).
	std::unordered_map<std::size_t, int> label_of;
	for ( std::size_t i = 0; i < selected.size(); ++i ) { label_of[selected[i]] = static_cast<int>( i ); }

	// parent_of_cluster: each cluster's parent cluster (from cluster-to-cluster edges).
	// point_cluster / point_lambda: the cluster a point fell out of, and at which lambda.
	std::unordered_map<std::size_t, std::size_t> parent_of_cluster;
	std::vector<std::size_t> point_cluster( n_points, N );
	std::vector<double> point_lambda( n_points, 0.0 );
	std::vector<char> point_seen( n_points, 0 );
	for ( const auto& e : tree ) {
		if ( e.child >= N ) {
			parent_of_cluster[e.child] = e.parent;
		} else {
			point_cluster[e.child] = e.parent;
			point_lambda[e.child] = e.lambda_val;
			point_seen[e.child] = 1;
		}
	}

	// Resolve each point to the nearest selected ancestor cluster.
	std::vector<int> final_label( n_points, -1 );
	for ( std::size_t p = 0; p < n_points; ++p ) {
		if ( !point_seen[p] ) { continue; }  // never fell out of any cluster => noise
		std::size_t c = point_cluster[p];
		int chosen = -1;
		while ( true ) {
			const auto sit = label_of.find( c );
			if ( sit != label_of.end() ) { chosen = sit->second; break; }
			const auto pit = parent_of_cluster.find( c );
			if ( pit == parent_of_cluster.end() ) { break; }  // reached the root without a selection
			c = pit->second;
		}
		final_label[p] = chosen;
		labels[p] = chosen;
	}

	// Per-final-cluster max lambda, then normalise each point's lambda into a [0,1] membership.
	std::vector<double> max_lambda( selected.size(), 0.0 );
	for ( std::size_t p = 0; p < n_points; ++p ) {
		if ( final_label[p] < 0 ) { continue; }
		const double lp = point_lambda[p];
		if ( std::isfinite( lp ) ) { max_lambda[static_cast<std::size_t>( final_label[p] )] = std::max( max_lambda[static_cast<std::size_t>( final_label[p] )], lp ); }
	}
	for ( std::size_t p = 0; p < n_points; ++p ) {
		if ( final_label[p] < 0 ) { continue; }
		const double ml = max_lambda[static_cast<std::size_t>( final_label[p] )];
		const double lp = point_lambda[p];
		probabilities[p] = ( ml > 0.0 && std::isfinite( lp ) ) ? std::min( lp, ml ) / ml : 1.0;
	}
}


//=== The MST-agnostic extractor (steps 4-6 packaged) ========================

// Turn any mutual-reachability MST over n points into a flat HDBSCAN* labelling: hierarchy ->
// condense -> stability -> EOM/leaf selection -> labels + probabilities. This is the part that
// does NOT care how the MST was built, so it is shared by exact hdbscan() (dense Prim) and the
// approximate shdbscan() (CEOs sparse graph). The caller has already noise-handled n < min_cluster_size.
// `weights` (optional) makes cluster sizes / stabilities count original-point weight (issue #46).
inline HdbscanResult extract_from_mst( const std::vector<MstEdge>& mst, std::size_t n,
									   std::size_t min_cluster_size, ClusterSelectionMethod method,
									   bool allow_single_cluster, const std::vector<std::size_t>& weights = {} ) {
	HdbscanResult result;
	const auto linkage = mst_to_linkage( mst, n, weights );
	const auto condensed = condense_tree( linkage, n, min_cluster_size, weights );
	const auto stability = compute_stabilities( condensed, n );
	const auto selected = ( method == ClusterSelectionMethod::Leaf )
		? select_leaf( condensed, stability, n, allow_single_cluster )
		: select_eom( condensed, stability, n, allow_single_cluster );
	do_labelling( condensed, selected, n, n, result.labels, result.probabilities );
	result.n_clusters = selected.size();
	return result;
}

}  // namespace detail


//=== Public entry point =====================================================

// Cluster `points` with HDBSCAN*. Returns per-point labels (-1 == noise) and membership
// probabilities (see HdbscanResult).
//   T, Dim                : coordinate type (float/double) and dimensionality (deduced).
//   Backend               : neighbor-search backend; defaults to nanoflann (must model KnnCoreDist).
//   min_cluster_size      : the smallest group of points HDBSCAN* will call a cluster (>= 2). The
//                           one parameter you really must choose; larger => fewer, bigger clusters.
//   min_samples           : the density smoother for the core distance (k). 0 (default) => use
//                           min_cluster_size. Larger => more points treated as noise, more
//                           conservative clusters. Self-inclusive (see the header CONVENTION NOTE).
//   method                : EOM (default, most persistent clusters) or Leaf (finest clusters).
//   allow_single_cluster  : if true, the whole cloud may be returned as one cluster; off by default
//                           so HDBSCAN* does not collapse everything into a single trivial cluster.
//   n_threads             : worker threads for the core-distance phase (0 => hardware concurrency).
//                           The MST and extraction are sequential.
//   dedup                 : (issue #46, ON by default) collapse bit-identical points to unique
//                           weighted points, cluster those, and expand the labels back to the
//                           ORIGINAL points -- the same partition, but the O(n^2) MST shrinks to the
//                           unique-point count (the big win for color/quantized data). Clouds with no
//                           duplicates fall through to the plain path, byte-for-byte unchanged. Set
//                           false to force the full cloud.
//   weights               : (issue #46) explicit per-point sample weights (sklearn `sample_weight`):
//                           each point counts as weights[i] originals for the core distance, cluster
//                           size and stability. Non-empty `weights` runs that weighting directly and
//                           BYPASSES dedup; empty (default) leaves dedup in charge. Requires a backend
//                           modeling KnnCoreDistWeighted.
//
//   mst_algo              : how the mutual-reachability MST is built (issue #66). DensePrim (default)
//                           is the O(n^2) complete-graph Prim -- exact, best up to n ~ 1e4. Boruvka
//                           is EXACT and sub-quadratic (same tree as DensePrim) -- the one for exact
//                           clustering past that ceiling. KnnGraph is a near-exact sub-quadratic
//                           k-NN-graph MST (requires KnnGraph<Backend>; falls back to DensePrim
//                           otherwise). See MstAlgorithm and docs/algorithms.md.
//
// Complexity: DensePrim is O(n^2) time / O(n) memory -- exact and simple, suited to small/medium n.
// MstAlgorithm::Boruvka is exact and sub-quadratic; MstAlgorithm::KnnGraph is near-exact and
// sub-quadratic; shdbscan() is the approximate-neighbor (cosine) path. The condensed-tree/extraction
// stages are MST-agnostic and shared across all of them.
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
HdbscanResult hdbscan( const std::vector<std::array<T, Dim>>& points, std::size_t min_cluster_size,
					   std::size_t min_samples = 0, ClusterSelectionMethod method = ClusterSelectionMethod::EOM,
					   bool allow_single_cluster = false, unsigned n_threads = 0, bool dedup = true,
					   const std::vector<std::size_t>& weights = {},
					   MstAlgorithm mst_algo = MstAlgorithm::DensePrim ) {
	static_assert( std::is_floating_point_v<T>, "hdbscan: coordinate type 'T' must be float or double" );
	static_assert( Dim >= 1, "hdbscan: dimension must be >= 1" );
	static_assert( NeighborSearch<Backend, T, Dim>, "Backend does not satisfy the NeighborSearch concept" );
	static_assert( KnnCoreDist<Backend, T, Dim>,
		"hdbscan requires a backend modeling KnnCoreDist (e.g. NanoflannBackend)" );

	if ( min_cluster_size < 2 ) { throw std::invalid_argument( "hdbscan: min_cluster_size must be >= 2" ); }
	if ( min_samples == 0 ) { min_samples = min_cluster_size; }

	const std::size_t n = points.size();
	HdbscanResult result;
	result.labels.assign( n, -1 );
	result.probabilities.assign( n, 0.0 );
	// Too few points to form even one cluster: everything is noise.
	if ( n < min_cluster_size ) { return result; }

	// Build the mutual-reachability MST for a (possibly weighted) cloud, dispatching on mst_algo. The
	// KnnGraph branch is only instantiated for backends that model it (if constexpr), so a backend
	// without knn_graph compiles fine and silently uses DensePrim. With DensePrim + empty weights this
	// is byte-for-byte the original path, keeping the pinned hdbscan: cases unchanged.
	const auto build_mst = [&]( const std::vector<std::array<T, Dim>>& pts, const std::vector<std::size_t>& w )
			-> std::vector<detail::MstEdge> {
		(void)mst_algo;  // unused when Backend lacks KnnGraph (the if constexpr below is discarded)
		if constexpr ( KnnGraph<Backend, T, Dim> ) {
			if ( mst_algo == MstAlgorithm::KnnGraph ) {
				return detail::knn_graph_mst<T, Dim, Backend>( pts, min_samples, n_threads, w );
			}
		}
		const auto core = detail::hdbscan_core_distances<T, Dim, Backend>( pts, min_samples, n_threads, w );
		if ( mst_algo == MstAlgorithm::Boruvka ) {
			// Exact Boruvka over the component-aware KD-tree: same tree as dense Prim, sub-quadratic.
			// Its BoruvkaEdge has the same (u, v, weight) layout as MstEdge; copy across.
			const auto bedges = detail::exact_mutual_reachability_mst( pts, core );
			std::vector<detail::MstEdge> mst;
			mst.reserve( bedges.size() );
			for ( const auto& e : bedges ) { mst.push_back( { e.u, e.v, e.weight } ); }
			return mst;
		}
		return detail::mutual_reachability_mst( pts, core );
	};

	// Explicit sample weights: weighted run on the points as given (labels parallel to the input).
	if ( !weights.empty() ) {
		if ( weights.size() != n ) { throw std::invalid_argument( "hdbscan: weights.size() must equal points.size()" ); }
		if constexpr ( KnnCoreDistWeighted<Backend, T, Dim> ) {
			const auto mst = build_mst( points, weights );
			return detail::extract_from_mst( mst, n, min_cluster_size, method, allow_single_cluster, weights );
		} else {
			throw std::invalid_argument( "hdbscan: explicit weights require a backend modeling KnnCoreDistWeighted" );
		}
	}

	// Auto-dedup: collapse bit-identical points, cluster the unique weighted cloud, expand to original.
	if constexpr ( KnnCoreDistWeighted<Backend, T, Dim> ) {
		if ( dedup ) {
			const auto d = deduplicate( points );
			if ( d.unique_points.size() < n ) {  // actual collapse => weighted path
				const std::size_t nu = d.unique_points.size();
				const auto mst = build_mst( d.unique_points, d.weights );
				const auto ur = detail::extract_from_mst( mst, nu, min_cluster_size, method, allow_single_cluster, d.weights );
				for ( std::size_t o = 0; o < n; ++o ) {
					const std::size_t u = d.unique_of_original[o];
					result.labels[o] = ur.labels[u];
					result.probabilities[o] = ur.probabilities[u];
				}
				result.n_clusters = ur.n_clusters;
				return result;
			}
			// no duplicates: nothing to gain; fall through to the plain (byte-identical) path.
		}
	}

	const auto mst = build_mst( points, {} );
	return detail::extract_from_mst( mst, n, min_cluster_size, method, allow_single_cluster );
}


//=== sHDBSCAN: scalable approximate HDBSCAN* via random projections ==========

// sHDBSCAN replaces the exact O(n^2) dense-Prim MST with an approximate mutual-reachability MST
// built from the same CEOs random-projection neighbor graph sOPTICS uses (see
// detail/random_projection.hpp), then runs the identical condense/stability/extract pipeline. It is
// to hdbscan() what sOPTICS is to OPTICS: a scalable, approximate variant whose output is randomized
// but deterministic in `seed` -- validate it by statistical agreement (Rand/NMI) with exact
// hdbscan(), not bit-identical labels.
//
// Metric is COSINE: points are L2-normalized onto the unit sphere internally, where Euclidean
// distance (what the mutual-reachability MST uses) is monotone in cosine distance -- so the
// clustering reflects cosine similarity, NOT raw Euclidean distance on the original data. L2 / L1
// embed into random Fourier features first (the same path as compute_soptics_reachability_dists),
// so the clustering then tracks Euclidean / Manhattan distance on the original data.
//
//   min_cluster_size / min_samples / method / allow_single_cluster : as in hdbscan(). min_samples
//                   also drives the CEOs candidate density (m defaults to 2*min_samples per vector).
//   epsilon       : generating distance on the unit sphere, in [0,2]; <= 0 => 2.0 (keep every CEOs
//                   candidate -- the pool is bounded regardless).
//   n_projections, k, m, seed : CEOs tunables (see compute_soptics_reachability_dists).
//   n_threads     : workers for the projection / candidate phases.
//   metric, kernel_scale, projection : as in compute_soptics_reachability_dists.
//   dedup         : (issue #46, ON by default) collapse bit-identical points to unique weighted
//                   points, cluster those, expand labels back to the originals. Exact dedup is
//                   lossless in any metric (identical coords are identical everywhere); for the
//                   cosine regime, merging by DIRECTION (scalar multiples) is an opt-in -- pass your
//                   own `weights` from deduplicate_cosine for that.
//   weights       : (issue #46) explicit per-point sample weights; non-empty BYPASSES dedup and runs
//                   the weighting directly (the approximate core distance and cluster sizes become
//                   weight-aware). Empty (default) leaves dedup in charge.
template <class T, std::size_t Dim>
HdbscanResult shdbscan( const std::vector<std::array<T, Dim>>& points, std::size_t min_cluster_size,
						std::size_t min_samples = 0, double epsilon = -1.0,
						unsigned n_projections = 1024, unsigned k = 0, std::size_t m = 0,
						unsigned seed = 42, unsigned n_threads = 0,
						ClusterSelectionMethod method = ClusterSelectionMethod::EOM,
						bool allow_single_cluster = false, Metric metric = Metric::Cosine,
						double kernel_scale = 0.0, SopticsProjection projection = SopticsProjection::Gaussian,
						bool dedup = true, const std::vector<std::size_t>& weights = {} ) {
	static_assert( std::is_floating_point_v<T>, "shdbscan: coordinate type 'T' must be float or double" );
	static_assert( Dim >= 1, "shdbscan: dimension must be >= 1" );

	if ( min_cluster_size < 2 ) { throw std::invalid_argument( "shdbscan: min_cluster_size must be >= 2" ); }
	if ( min_samples == 0 ) { min_samples = min_cluster_size; }

	const std::size_t n = points.size();
	HdbscanResult result;
	result.labels.assign( n, -1 );
	result.probabilities.assign( n, 0.0 );
	if ( n < min_cluster_size ) { return result; }

	const bool explicit_weights = !weights.empty();
	if ( explicit_weights && weights.size() != n ) {
		throw std::invalid_argument( "shdbscan: weights.size() must equal points.size()" );
	}

	// Auto-dedup on the ORIGINAL points (before any embedding/normalization), then recurse weighted
	// and expand. Only when the caller did not pass explicit weights.
	if ( !explicit_weights && dedup ) {
		const auto d = deduplicate( points );
		if ( d.unique_points.size() < n ) {
			const auto ur = shdbscan<T, Dim>( d.unique_points, min_cluster_size, min_samples, epsilon,
											  n_projections, k, m, seed, n_threads, method, allow_single_cluster,
											  metric, kernel_scale, projection, false, d.weights );
			for ( std::size_t o = 0; o < n; ++o ) {
				const std::size_t u = d.unique_of_original[o];
				result.labels[o] = ur.labels[u];
				result.probabilities[o] = ur.probabilities[u];
			}
			result.n_clusters = ur.n_clusters;
			return result;
		}
	}

	// Non-cosine metric: embed into random Fourier features (cosine-on-features ~ target kernel),
	// then recurse on the cosine pipeline (carrying weights). Indices line up 1:1, labels map back.
	if ( metric != Metric::Cosine ) {
		constexpr std::size_t FeatDim = 256;
		const double sigma = ( kernel_scale > 0.0 ) ? kernel_scale : detail::auto_kernel_scale( points, metric );
		const auto feats = detail::embed_random_features<FeatDim, T, Dim>( points, metric, sigma, seed, n_threads );
		return shdbscan<double, FeatDim>( feats, min_cluster_size, min_samples, epsilon, n_projections, k, m,
										  seed, n_threads, method, allow_single_cluster, Metric::Cosine, 0.0,
										  projection, false, weights );
	}

	// L2-normalize onto the unit sphere (cosine metric); the origin has no direction, leave it.
	std::vector<std::array<T, Dim>> unit( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		double nrm_sq = 0.0;
		for ( std::size_t c = 0; c < Dim; ++c ) { const double v = static_cast<double>( points[i][c] ); nrm_sq += v * v; }
		const double nrm = std::sqrt( nrm_sq );
		if ( nrm > 0.0 ) {
			for ( std::size_t c = 0; c < Dim; ++c ) { unit[i][c] = static_cast<T>( static_cast<double>( points[i][c] ) / nrm ); }
		} else {
			unit[i] = points[i];
		}
	}

	const double eps = ( epsilon <= 0.0 ) ? 2.0 : epsilon;
	detail::CeosParams params;
	params.n_projections = n_projections;
	params.k = k;
	params.m = m;
	params.seed = seed;
	params.n_threads = n_threads;
	params.projection = ( projection == SopticsProjection::Structured )
		? detail::CeosParams::Projection::Structured
		: detail::CeosParams::Projection::Gaussian;

	// CEOs candidate graph (symmetric, self-inclusive) + each candidate's squared distance.
	std::vector<std::vector<double>> neighbor_sq;
	const auto neighbors = detail::ceos_neighbors( unit, eps, min_samples, params, &neighbor_sq );

	const auto core = explicit_weights
		? detail::approx_core_distances( neighbor_sq, min_samples, neighbors, weights )
		: detail::approx_core_distances( neighbor_sq, min_samples );
	const auto mst = detail::approx_mutual_reachability_mst( neighbors, neighbor_sq, core, n );
	return detail::extract_from_mst( mst, n, min_cluster_size, method, allow_single_cluster, weights );
}

}  // namespace optics
