// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Exact, sub-quadratic mutual-reachability MST for HDBSCAN* (issue #66, Phase 2).
//
// Where the dense-Prim MST is O(n^2) over the COMPLETE graph and the k-NN-graph MST is near-exact,
// this builds the SAME minimum spanning tree as dense Prim -- exact (identical total weight) -- but
// sub-quadratically, with a Boruvka algorithm accelerated by a purpose-built KD-tree. It is the
// exact-leaning end of the #66 spectrum; the k-NN-graph path remains the cheaper approximate option.
//
// Why a dedicated tree? The neighbor-search backend (nanoflann) exposes neither node bounding boxes
// nor a traversal API, and Boruvka needs both: per round it asks "for each component, what is the
// cheapest mutual-reachability edge to a DIFFERENT component?", which a spatial tree answers by
// pruning whole subtrees. So this header carries a compact KD-tree that stores, per node, a bounding
// box, the minimum core distance under it, and (refreshed each round) the component its points belong
// to (or MIXED). The traversal is single-tree (each point searched against the tree) with two prunes
// -- a node fully inside the query's own component is skipped, and a node whose best-possible
// mutual-reachability lower bound cannot beat the component's current best is skipped. (A dual-tree
// node-vs-node refinement and parallel rounds are further speed-ups, not needed for correctness.)
//
// Ideas drawn from TutteInstitute/fast_hdbscan (component_aware_query_recursion / merge_components /
// update_component_vectors) and the March/Ram/Gray "Fast EMST" dual-tree Boruvka; reimplemented from
// the algorithm to stay MIT-clean.

#include "math.hpp"  // detail::square_dist

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace optics {
namespace detail {

// One undirected MST edge (u, v at mutual-reachability weight). A standalone type so this header does
// not depend on hdbscan.hpp's MstEdge; the caller copies the fields across (identical layout).
struct BoruvkaEdge {
	std::size_t u = 0;
	std::size_t v = 0;
	double weight = 0.0;
};


// Compact KD-tree specialised for Boruvka: bounding boxes + per-node min core distance + a per-node
// component tag refreshed each round. Built once over the points; geometry never changes, only the
// component tags. Points are referenced through a reordered index array (order_), so leaves are
// contiguous ranges into it.
template <class T, std::size_t Dim>
class BoruvkaKdTree {
public:
	static constexpr std::size_t MIXED = std::numeric_limits<std::size_t>::max();

	struct Node {
		std::array<T, Dim> lo{};   // bounding-box corner (min per dim)
		std::array<T, Dim> hi{};   // bounding-box corner (max per dim)
		double min_core = 0.0;     // smallest core distance among the points under this node
		std::size_t comp = MIXED;  // the component all points under it share, or MIXED if they differ
		std::size_t start = 0;     // [start, start+count) into order_ -- the points under this node
		std::size_t count = 0;
		int left = -1;             // child node indices; -1/-1 => leaf
		int right = -1;
		bool is_leaf() const { return left < 0; }
	};

	BoruvkaKdTree( const std::vector<std::array<T, Dim>>& pts, std::size_t leaf_size )
			: pts_( &pts ), order_( pts.size() ) {
		std::iota( order_.begin(), order_.end(), std::size_t( 0 ) );
		if ( !pts.empty() ) {
			nodes_.reserve( 2 * pts.size() );
			root_ = build( 0, pts.size(), std::max<std::size_t>( 1, leaf_size ) );
		}
	}

	int root() const { return root_; }
	const std::vector<Node>& nodes() const { return nodes_; }
	const std::vector<std::size_t>& order() const { return order_; }

	// Fill each node's min_core bottom-up (call once, after core distances are known).
	void set_core( const std::vector<double>& core ) {
		if ( root_ >= 0 ) { fill_min_core( root_, core ); }
	}

	// Refresh each node's component tag from the current per-point components (call once per round).
	void refresh_components( const std::vector<std::size_t>& point_comp ) {
		if ( root_ >= 0 ) { fill_components( root_, point_comp ); }
	}

	// Squared distance from point q to the nearest corner of node `id`'s bounding box (0 if inside).
	double box_min_dist_sq( int id, const std::array<T, Dim>& q ) const {
		const Node& nd = nodes_[static_cast<std::size_t>( id )];
		double s = 0.0;
		for ( std::size_t d = 0; d < Dim; ++d ) {
			const double v = static_cast<double>( q[d] );
			const double lo = static_cast<double>( nd.lo[d] );
			const double hi = static_cast<double>( nd.hi[d] );
			const double diff = ( v < lo ) ? ( lo - v ) : ( ( v > hi ) ? ( v - hi ) : 0.0 );
			s += diff * diff;
		}
		return s;
	}

private:
	int build( std::size_t start, std::size_t count, std::size_t leaf_size ) {
		const int id = static_cast<int>( nodes_.size() );
		nodes_.push_back( Node{} );  // reserve this node's slot before recursing into children
		Node nd;
		nd.start = start;
		nd.count = count;

		// Bounding box over this range.
		nd.lo = ( *pts_ )[order_[start]];
		nd.hi = nd.lo;
		for ( std::size_t i = start + 1; i < start + count; ++i ) {
			const auto& p = ( *pts_ )[order_[i]];
			for ( std::size_t d = 0; d < Dim; ++d ) {
				if ( p[d] < nd.lo[d] ) { nd.lo[d] = p[d]; }
				if ( p[d] > nd.hi[d] ) { nd.hi[d] = p[d]; }
			}
		}

		if ( count <= leaf_size ) {
			nodes_[static_cast<std::size_t>( id )] = nd;  // leaf
			return id;
		}

		// Split the widest dimension at the median (nth_element on the index range).
		std::size_t axis = 0;
		double widest = -1.0;
		for ( std::size_t d = 0; d < Dim; ++d ) {
			const double extent = static_cast<double>( nd.hi[d] ) - static_cast<double>( nd.lo[d] );
			if ( extent > widest ) { widest = extent; axis = d; }
		}
		const std::size_t mid = start + count / 2;
		std::nth_element( order_.begin() + start, order_.begin() + mid, order_.begin() + start + count,
						  [&]( std::size_t a, std::size_t b ) { return ( *pts_ )[a][axis] < ( *pts_ )[b][axis]; } );

		nd.left = build( start, mid - start, leaf_size );
		nd.right = build( mid, start + count - mid, leaf_size );
		nodes_[static_cast<std::size_t>( id )] = nd;  // assign after children (the slot at id is stable)
		return id;
	}

	double fill_min_core( int id, const std::vector<double>& core ) {
		Node& nd = nodes_[static_cast<std::size_t>( id )];
		if ( nd.is_leaf() ) {
			double m = std::numeric_limits<double>::infinity();
			for ( std::size_t i = nd.start; i < nd.start + nd.count; ++i ) { m = std::min( m, core[order_[i]] ); }
			nd.min_core = m;
			return m;
		}
		const double m = std::min( fill_min_core( nd.left, core ), fill_min_core( nd.right, core ) );
		nd.min_core = m;
		return m;
	}

	std::size_t fill_components( int id, const std::vector<std::size_t>& point_comp ) {
		Node& nd = nodes_[static_cast<std::size_t>( id )];
		if ( nd.is_leaf() ) {
			std::size_t c = point_comp[order_[nd.start]];
			for ( std::size_t i = nd.start + 1; i < nd.start + nd.count; ++i ) {
				if ( point_comp[order_[i]] != c ) { c = MIXED; break; }
			}
			nd.comp = c;
			return c;
		}
		const std::size_t lc = fill_components( nd.left, point_comp );
		const std::size_t rc = fill_components( nd.right, point_comp );
		nd.comp = ( lc == rc && lc != MIXED ) ? lc : MIXED;
		return nd.comp;
	}

	const std::vector<std::array<T, Dim>>* pts_;
	std::vector<std::size_t> order_;
	std::vector<Node> nodes_;
	int root_ = -1;
};


// Recursive single-tree search: find, for query point q in component cq, the cheapest
// mutual-reachability edge to a DIFFERENT component, tightening (best_w, best_u, best_v) -- the
// component's current best, shared across its points so the bound tightens as the round proceeds.
template <class T, std::size_t Dim>
void boruvka_search( const BoruvkaKdTree<T, Dim>& tree, int id,
					 const std::vector<std::array<T, Dim>>& pts, const std::vector<double>& core,
					 std::size_t q, std::size_t cq, const std::vector<std::size_t>& point_comp,
					 double& best_w, std::size_t& best_u, std::size_t& best_v ) {
	const auto& nd = tree.nodes()[static_cast<std::size_t>( id )];
	if ( nd.comp == cq ) { return; }  // whole subtree is q's own component -- no valid target

	// Lower bound on mutual reachability from q to ANY point under this node:
	//   max(core(q), min_core_under_node, box_distance) <= max(core(q), core(t), d(q,t)) = d_mreach.
	const double d_lo = std::sqrt( tree.box_min_dist_sq( id, pts[q] ) );
	const double lb = std::max( core[q], std::max( d_lo, nd.min_core ) );
	if ( lb >= best_w ) { return; }  // cannot beat the component's current best

	if ( nd.is_leaf() ) {
		const auto& order = tree.order();
		for ( std::size_t i = nd.start; i < nd.start + nd.count; ++i ) {
			const std::size_t t = order[i];
			if ( point_comp[t] == cq ) { continue; }  // same component (includes q itself)
			const double d = std::sqrt( square_dist( pts[q], pts[t] ) );
			const double w = std::max( core[q], std::max( core[t], d ) );
			if ( w < best_w ) { best_w = w; best_u = q; best_v = t; }
		}
		return;
	}

	// Descend the nearer child first so the bound tightens before the farther child is considered.
	const double dl = tree.box_min_dist_sq( nd.left, pts[q] );
	const double dr = tree.box_min_dist_sq( nd.right, pts[q] );
	const int first = ( dl <= dr ) ? nd.left : nd.right;
	const int second = ( dl <= dr ) ? nd.right : nd.left;
	boruvka_search( tree, first, pts, core, q, cq, point_comp, best_w, best_u, best_v );
	boruvka_search( tree, second, pts, core, q, cq, point_comp, best_w, best_u, best_v );
}


// Exact mutual-reachability MST via Boruvka over the component-aware KD-tree. Produces the same tree
// (identical total weight) as dense Prim, sub-quadratically. The complete mutual-reachability graph is
// always connected, so this terminates with exactly n-1 edges -- no connectivity fix-up (unlike the
// sparse k-NN / CEOs paths). The round loop is sequential (parallelising it is future work); the core
// distances are computed in parallel upstream.
template <class T, std::size_t Dim>
std::vector<BoruvkaEdge> exact_mutual_reachability_mst( const std::vector<std::array<T, Dim>>& points,
														const std::vector<double>& core,
														std::size_t leaf_size = 16 ) {
	const std::size_t n = points.size();
	std::vector<BoruvkaEdge> mst;
	if ( n < 2 ) { return mst; }

	BoruvkaKdTree<T, Dim> tree( points, leaf_size );
	tree.set_core( core );

	// Union-find (path-halving + union by size) over the points; roots are the component ids.
	std::vector<std::size_t> uf( n ), sz( n, 1 );
	std::iota( uf.begin(), uf.end(), std::size_t( 0 ) );
	const auto find = [&]( std::size_t x ) {
		while ( uf[x] != x ) { uf[x] = uf[uf[x]]; x = uf[x]; }
		return x;
	};

	std::vector<std::size_t> point_comp( n );
	std::vector<double> comp_best_w( n );
	std::vector<std::size_t> comp_best_u( n ), comp_best_v( n );
	constexpr double INF = std::numeric_limits<double>::infinity();

	mst.reserve( n - 1 );
	std::size_t num_components = n;
	while ( num_components > 1 ) {
		for ( std::size_t i = 0; i < n; ++i ) { point_comp[i] = find( i ); }
		tree.refresh_components( point_comp );
		std::fill( comp_best_w.begin(), comp_best_w.end(), INF );

		// Each point finds its component's cheapest cross-edge; the shared per-component bound
		// (comp_best_w[c]) tightens across points, so later points in a component prune more.
		for ( std::size_t q = 0; q < n; ++q ) {
			const std::size_t c = point_comp[q];
			boruvka_search( tree, tree.root(), points, core, q, c, point_comp,
							comp_best_w[c], comp_best_u[c], comp_best_v[c] );
		}

		// Add each component's best edge and union (a component is identified by its root index c).
		for ( std::size_t c = 0; c < n; ++c ) {
			if ( point_comp[c] != c || comp_best_w[c] >= INF ) { continue; }  // not a root / no edge
			const std::size_t u = comp_best_u[c], v = comp_best_v[c];
			std::size_t ru = find( u ), rv = find( v );
			if ( ru == rv ) { continue; }  // already merged this round (the other endpoint picked us)
			if ( sz[ru] < sz[rv] ) { std::swap( ru, rv ); }
			uf[rv] = ru;
			sz[ru] += sz[rv];
			mst.push_back( { u, v, comp_best_w[c] } );
			--num_components;
		}
	}
	return mst;
}

}  // namespace detail
}  // namespace optics
