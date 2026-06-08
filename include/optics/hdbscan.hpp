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
#include "detail/math.hpp"
#include "detail/thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
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
template <class T, std::size_t Dim, class Backend>
std::vector<double> hdbscan_core_distances( const std::vector<std::array<T, Dim>>& points,
											 std::size_t min_samples, unsigned n_threads ) {
	static_assert( KnnCoreDist<Backend, T, Dim>,
		"hdbscan requires a backend modeling KnnCoreDist (e.g. NanoflannBackend)" );
	const std::size_t n = points.size();
	std::vector<double> core( n, 0.0 );
	const Backend backend( points );
	const T no_cap = std::numeric_limits<T>::max();  // no radius cap: a plain k-NN distance
	detail::parallel_for( n_threads, n, [&]( std::size_t i ) {
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


//=== Step 4: single-linkage hierarchy from the MST ==========================

// Union-find that mints a fresh node id on every merge (the dendrogram's internal nodes). find()
// returns the id of the current top-most node containing x; do_union() joins two such top nodes
// under a brand-new id. This is the standard HDBSCAN single-linkage labelling.
struct LinkageUnionFind {
	std::vector<std::size_t> parent;  // size 2N-1; leaves 0..N-1, internal nodes N..2N-2
	std::vector<std::size_t> size;
	std::size_t next_label;

	explicit LinkageUnionFind( std::size_t N ) : parent( 2 * N - 1 ), size( 2 * N - 1, 1 ), next_label( N ) {
		for ( std::size_t i = 0; i < parent.size(); ++i ) { parent[i] = i; }
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
// rows of the single-linkage dendrogram. The final merge (node 2N-2) is the root.
inline std::vector<LinkageRow> mst_to_linkage( std::vector<MstEdge> edges, std::size_t N ) {
	std::vector<LinkageRow> rows;
	if ( N < 2 ) { return rows; }
	std::sort( edges.begin(), edges.end(), []( const MstEdge& a, const MstEdge& b ) { return a.weight < b.weight; } );
	LinkageUnionFind uf( N );
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

// Number of original points under a linkage node: 1 for a leaf (id < N), else the merge's size.
inline std::size_t linkage_node_size( const std::vector<LinkageRow>& rows, std::size_t N, std::size_t node ) {
	return node < N ? std::size_t( 1 ) : rows[node - N].size;
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
												 std::size_t min_cluster_size ) {
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

	for ( const std::size_t node : bfs_linkage( rows, N, root ) ) {
		if ( ignore[node] || node < N ) { continue; }
		const LinkageRow& row = rows[node - N];
		const std::size_t left = row.left;
		const std::size_t right = row.right;
		const double lambda_value = lambda_of( row.dist );
		const std::size_t left_count = linkage_node_size( rows, N, left );
		const std::size_t right_count = linkage_node_size( rows, N, right );

		if ( left_count >= min_cluster_size && right_count >= min_cluster_size ) {
			// Genuine split: both sides become new clusters.
			relabel[left] = next_label++;
			result.push_back( { relabel[node], relabel[left], lambda_value, left_count } );
			relabel[right] = next_label++;
			result.push_back( { relabel[node], relabel[right], lambda_value, right_count } );
		} else if ( left_count < min_cluster_size && right_count < min_cluster_size ) {
			// Both sides too small: every point falls out of the current cluster here.
			for ( const std::size_t sub : bfs_linkage( rows, N, left ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, 1 } ); }
				ignore[sub] = 1;
			}
			for ( const std::size_t sub : bfs_linkage( rows, N, right ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, 1 } ); }
				ignore[sub] = 1;
			}
		} else if ( left_count < min_cluster_size ) {
			// Left too small falls out; right continues as the SAME cluster.
			relabel[right] = relabel[node];
			for ( const std::size_t sub : bfs_linkage( rows, N, left ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, 1 } ); }
				ignore[sub] = 1;
			}
		} else {
			// Right too small falls out; left continues as the SAME cluster.
			relabel[left] = relabel[node];
			for ( const std::size_t sub : bfs_linkage( rows, N, right ) ) {
				if ( sub < N ) { result.push_back( { relabel[node], sub, lambda_value, 1 } ); }
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
//
// Complexity: the dense-Prim MST is O(n^2) time / O(n) memory -- exact and simple, suited to
// small/medium n. A sub-quadratic MST (Boruvka, or the sOPTICS graph) is future work (issue #52
// follow-up); the condensed-tree/extraction stages are already MST-agnostic and would be reused.
template <class T, std::size_t Dim, class Backend = NanoflannBackend<T, Dim>>
HdbscanResult hdbscan( const std::vector<std::array<T, Dim>>& points, std::size_t min_cluster_size,
					   std::size_t min_samples = 0, ClusterSelectionMethod method = ClusterSelectionMethod::EOM,
					   bool allow_single_cluster = false, unsigned n_threads = 0 ) {
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

	const auto core = detail::hdbscan_core_distances<T, Dim, Backend>( points, min_samples, n_threads );
	const auto mst = detail::mutual_reachability_mst( points, core );
	const auto linkage = detail::mst_to_linkage( mst, n );
	const auto condensed = detail::condense_tree( linkage, n, min_cluster_size );
	const auto stability = detail::compute_stabilities( condensed, n );

	const auto selected = ( method == ClusterSelectionMethod::Leaf )
		? detail::select_leaf( condensed, stability, n, allow_single_cluster )
		: detail::select_eom( condensed, stability, n, allow_single_cluster );

	detail::do_labelling( condensed, selected, n, n, result.labels, result.probabilities );
	result.n_clusters = selected.size();
	return result;
}

}  // namespace optics
