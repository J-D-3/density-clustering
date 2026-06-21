// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
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
// mutual-reachability lower bound cannot beat the component's current best is skipped.
//
// Each Boruvka round runs in PARALLEL: the query points are split into per-worker blocks, each worker
// keeps a private per-component best (so writes are unshared and scheduling-independent), and the
// blocks are merged afterwards -- exact at any thread count, deterministic for a fixed one (see
// exact_mutual_reachability_mst). A leaf-batched DUAL-TREE variant (recurse a whole query leaf at
// once) was implemented and benchmarked but REVERTED: a mixed-component query leaf must prune by its
// loosest point's bound, and early rounds are almost all mixed, so it pruned worse than the per-point
// search and ran slower. The per-component bound, not query batching, is what makes this fast.
//
// Ideas drawn from TutteInstitute/fast_hdbscan (component_aware_query_recursion / merge_components /
// update_component_vectors) and the March/Ram/Gray "Fast EMST" dual-tree Boruvka; reimplemented from
// the algorithm to stay MIT-clean.

#include "math.hpp"         // detail::square_dist
#include "thread_pool.hpp"  // detail::parallel_for (parallel rounds)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

// Opt-in per-round profiling (build any consumer with -DOPTICS_BORUVKA_PROFILE / /DOPTICS_BORUVKA_PROFILE):
// prints each round's component count and parallel-search time to stderr, so the per-round cost
// distribution is visible (e.g. whether early or late rounds dominate). Zero overhead when undefined.
#ifdef OPTICS_BORUVKA_PROFILE
#include <chrono>
#include <cstdio>
#endif

namespace optics {
namespace detail {
#ifdef OPTICS_BORUVKA_PROFILE
// Pruning-efficiency counters (single-thread profiling only): how many tree nodes the search touches
// and how many point-point distances it evaluates. The ratio of dist-evals to n exposes whether the
// KD-tree's box bound still prunes in high dimension.
//
// #76 INVESTIGATION (closed, negative): these counters confirmed the box bound's pruning collapses with
// dimension -- ~3 nodes/query in 3-D vs ~82 in 16-D (a ~27x rise). BUT a prototype tightening the prune
// with a per-node centroid+radius BALL bound (max(box, ball), exact -- identical MST weight) recovered
// little: measured (optics_hdbscan_boruvka_ball_probe) only ~1.1-1.16x at 24-32 D and a REGRESSION
// below 24-D (the O(Dim) ball-distance cost per node exceeds the pruning saved). That does not close the
// ~2.8x gap to the near-exact KnnGraph backbone, which Auto already picks for dim >= 16 (Rand 1.0), so a
// full ball/cover tree would not change the policy. The high-dim regime is better served by KnnGraph and
// the HNSW-KnnGraph source (#73). The ball prototype was therefore reverted; this note records the result.
inline std::size_t g_boruvka_nodes_visited = 0;
inline std::size_t g_boruvka_dist_evals = 0;
#endif
}  // namespace detail
}  // namespace optics

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
	const std::vector<int>& leaves() const { return leaves_; }  // leaf node ids (query side of the dual-tree)

	// Squared distance between the bounding boxes of nodes a and b (0 if they overlap); the lower bound
	// on the Euclidean distance between ANY point under a and ANY point under b. Used by the dual-tree
	// search to prune a whole query-leaf-vs-target-node pair at once.
	double box_box_min_dist_sq( int a, int b ) const {
		const Node& na = nodes_[static_cast<std::size_t>( a )];
		const Node& nb = nodes_[static_cast<std::size_t>( b )];
		double s = 0.0;
		for ( std::size_t d = 0; d < Dim; ++d ) {
			const double alo = static_cast<double>( na.lo[d] ), ahi = static_cast<double>( na.hi[d] );
			const double blo = static_cast<double>( nb.lo[d] ), bhi = static_cast<double>( nb.hi[d] );
			const double gap = ( ahi < blo ) ? ( blo - ahi ) : ( ( bhi < alo ) ? ( alo - bhi ) : 0.0 );
			s += gap * gap;
		}
		return s;
	}

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
			leaves_.push_back( id );                      // remember leaves for the dual-tree query pass
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
	std::vector<int> leaves_;
	int root_ = -1;
};


// Single-tree search: improve query point q's COMPONENT (cq) cheapest mutual-reachability edge to a
// different component, by recursing q against the target tree. Pruning is by the per-component best
// (best_w/best_u/best_v passed as the component's slot), which tightens as edges are found -- both
// within q's search and across earlier points the same worker processed in component cq. These are the
// WORKER's private component bests (each worker owns a contiguous block of query points, so its writes
// are unshared); they are merged across workers afterwards.
//
// A leaf-batched dual-tree variant was tried (recurse a whole query leaf at once) but measured SLOWER:
// a mixed-component leaf must prune by its loosest point's bound, and in early rounds almost every leaf
// is mixed, so the per-point bound here prunes harder than the descent-amortisation saves.
template <class T, std::size_t Dim>
void boruvka_search( const BoruvkaKdTree<T, Dim>& tree, int id,
					 const std::vector<std::array<T, Dim>>& pts, const std::vector<double>& core,
					 std::size_t q, std::size_t cq, const std::vector<std::size_t>& point_comp,
					 double& best_w, std::size_t& best_u, std::size_t& best_v ) {
	const auto& nd = tree.nodes()[static_cast<std::size_t>( id )];
#ifdef OPTICS_BORUVKA_PROFILE
	++g_boruvka_nodes_visited;
#endif
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
#ifdef OPTICS_BORUVKA_PROFILE
			++g_boruvka_dist_evals;
#endif
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


// Leaf-batched DUAL-TREE search: improve component cq's cheapest cross-component edge using a whole
// PURE query leaf (qleaf, all its points in cq) at once, recursing it against the target tree. This is
// the late-round counterpart to the per-point boruvka_search: when components are few and large, query
// leaves are mostly pure, so one descent amortises across all the leaf's points and the box-vs-box
// bound prunes whole target subtrees that no point in the leaf could beat. (Early rounds are the
// opposite -- almost every leaf is mixed -- which is why the original blanket dual-tree was reverted;
// here it is gated to late rounds by num_components, see exact_mutual_reachability_mst.)
template <class T, std::size_t Dim>
void boruvka_dual_search( const BoruvkaKdTree<T, Dim>& tree, int target_id, int qleaf,
						  const std::vector<std::array<T, Dim>>& pts, const std::vector<double>& core,
						  std::size_t cq, const std::vector<std::size_t>& point_comp,
						  double& best_w, std::size_t& best_u, std::size_t& best_v ) {
	const auto& tn = tree.nodes()[static_cast<std::size_t>( target_id )];
#ifdef OPTICS_BORUVKA_PROFILE
	++g_boruvka_nodes_visited;
#endif
	if ( tn.comp == cq ) { return; }  // whole target subtree is the query leaf's own component

	const auto& qn = tree.nodes()[static_cast<std::size_t>( qleaf )];
	// Lower bound on mutual reachability between ANY point in the query leaf and ANY under the target:
	//   max( min_core(query leaf), min_core(target), boxdist(qbox, tbox) ) <= max(core(q),core(t),d).
	const double bd = std::sqrt( tree.box_box_min_dist_sq( qleaf, target_id ) );
	const double lb = std::max( std::max( qn.min_core, tn.min_core ), bd );
	if ( lb >= best_w ) { return; }  // no pair across these boxes can beat the component's current best

	if ( tn.is_leaf() ) {
		const auto& order = tree.order();
		for ( std::size_t i = qn.start; i < qn.start + qn.count; ++i ) {
			const std::size_t q = order[i];
			for ( std::size_t j = tn.start; j < tn.start + tn.count; ++j ) {
				const std::size_t t = order[j];
				if ( point_comp[t] == cq ) { continue; }  // same component (includes the leaf's own points)
#ifdef OPTICS_BORUVKA_PROFILE
				++g_boruvka_dist_evals;
#endif
				const double d = std::sqrt( square_dist( pts[q], pts[t] ) );
				const double w = std::max( core[q], std::max( core[t], d ) );
				if ( w < best_w ) { best_w = w; best_u = q; best_v = t; }
			}
		}
		return;
	}

	// Descend the nearer target child first (by box-vs-box distance to the query leaf) so the bound
	// tightens before the farther child is considered.
	const double dl = tree.box_box_min_dist_sq( tn.left, qleaf );
	const double dr = tree.box_box_min_dist_sq( tn.right, qleaf );
	const int first = ( dl <= dr ) ? tn.left : tn.right;
	const int second = ( dl <= dr ) ? tn.right : tn.left;
	boruvka_dual_search( tree, first, qleaf, pts, core, cq, point_comp, best_w, best_u, best_v );
	boruvka_dual_search( tree, second, qleaf, pts, core, cq, point_comp, best_w, best_u, best_v );
}


// Exact mutual-reachability MST via Boruvka over the component-aware KD-tree. Produces the same tree
// (identical total weight) as dense Prim, sub-quadratically. The complete mutual-reachability graph is
// always connected, so this terminates with exactly n-1 edges -- no connectivity fix-up (unlike the
// sparse k-NN / CEOs paths).
//
// Each round partitions the query POINTS into `n_workers` contiguous blocks (n_threads, 0 => hardware
// concurrency) and runs them in PARALLEL: each worker keeps its OWN per-component bests, so its writes
// are unshared (race-free) and -- because the point partition and per-worker processing order are
// fixed -- independent of scheduling. The per-worker bests are then MERGED (min weight, ties by worker
// then point index) into the round's per-component edges, which are added and unioned -- the standard
// Boruvka step. Keeping the component bound per worker preserves the strong per-component pruning that
// makes the search fast, while still parallelising. The result is deterministic for a fixed
// n_threads; under mutual-reachability ties (equal-weight cross-edges, e.g. on core-distance plateaus)
// the specific edge chosen can vary with thread count, but the MST total weight is invariant (still
// exact). Core distances were computed in parallel upstream.
//
// ROUND-ADAPTIVE traversal (issue #75): the per-round profile shows the LATE rounds (few, large
// components) dominate the cost -- there per-point search is expensive because each of a big
// component's points independently scans for its cheapest cross-component edge. Those rounds also have
// mostly-PURE query leaves, so a leaf-batched dual-tree (boruvka_dual_search) amortises one descent
// across a whole leaf and prunes by box-vs-box. We therefore switch to the dual-tree once
// num_components <= `dual_max_components`; early rounds (many tiny, mixed components -- where the
// blanket dual-tree was reverted as slower) stay on per-point search. The switch is exact either way
// (same total MST weight). `dual_max_components`: 0 disables the dual path entirely (pure per-point,
// the pre-#75 behaviour); the sentinel `kBoruvkaAutoDual` (default) auto-picks n / (2*leaf_size) --
// roughly "the average component spans >= 2 leaves, so leaves are mostly pure".
//
// DIMENSION GATE: the dual-tree's box-vs-box bound loosens with dimension (curse of dimensionality), so
// it only prunes -- and only pays off -- in LOW dimension. Measured (optics_hdbscan_boruvka_dual_probe,
// 4 threads, blobs): ~1.1-1.5x at 2-6 D, but 0.87x at 8 D and 0.25-0.40x at 16 D. The dual path is
// therefore hard-gated to Dim <= kBoruvkaDualMaxDim at compile time and is NEVER taken above it,
// regardless of dual_max_components -- so Auto (#72 picks Boruvka up to 12 D) and any explicit caller
// stay safe from the high-dim regression. Per-point search remains the universal exact path.
inline constexpr std::size_t kBoruvkaAutoDual = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kBoruvkaDualMaxDim = 6;

template <class T, std::size_t Dim>
std::vector<BoruvkaEdge> exact_mutual_reachability_mst( const std::vector<std::array<T, Dim>>& points,
														const std::vector<double>& core,
														unsigned n_threads = 0, std::size_t leaf_size = 16,
														std::size_t dual_max_components = kBoruvkaAutoDual ) {
	const std::size_t n = points.size();
	std::vector<BoruvkaEdge> mst;
	if ( n < 2 ) { return mst; }

	BoruvkaKdTree<T, Dim> tree( points, leaf_size );
	tree.set_core( core );

	// Resolve the auto sentinel to the heuristic threshold (mostly-pure leaves => dual-tree pays off).
	// [[maybe_unused]]: only read inside the Dim <= kBoruvkaDualMaxDim branch below.
	[[maybe_unused]] const std::size_t dual_threshold = ( dual_max_components == kBoruvkaAutoDual )
		? n / ( 2 * std::max<std::size_t>( 1, leaf_size ) )
		: dual_max_components;

	// Union-find (path-halving + union by size) over the points; roots are the component ids.
	std::vector<std::size_t> uf( n ), sz( n, 1 );
	std::iota( uf.begin(), uf.end(), std::size_t( 0 ) );
	const auto find = [&]( std::size_t x ) {
		while ( uf[x] != x ) { uf[x] = uf[uf[x]]; x = uf[x]; }
		return x;
	};

	// One private (component-best) workspace per worker; partition the query points into blocks.
	const unsigned n_workers = std::max( 1u, std::min<unsigned>( detail::resolve_thread_count( n_threads ),
		static_cast<unsigned>( n ) ) );
	constexpr double INF = std::numeric_limits<double>::infinity();
	std::vector<std::vector<double>> wbest_w( n_workers, std::vector<double>( n, INF ) );
	std::vector<std::vector<std::size_t>> wbest_u( n_workers, std::vector<std::size_t>( n, 0 ) );
	std::vector<std::vector<std::size_t>> wbest_v( n_workers, std::vector<std::size_t>( n, 0 ) );
	const std::size_t pblock = ( n + n_workers - 1 ) / n_workers;

	std::vector<std::size_t> point_comp( n );
	std::vector<double> comp_best_w( n );
	std::vector<std::size_t> comp_best_u( n ), comp_best_v( n );

	mst.reserve( n - 1 );
	std::size_t num_components = n;
#ifdef OPTICS_BORUVKA_PROFILE
	std::size_t profile_round = 0;
#endif
	while ( num_components > 1 ) {
		for ( std::size_t i = 0; i < n; ++i ) { point_comp[i] = find( i ); }
		tree.refresh_components( point_comp );

#ifdef OPTICS_BORUVKA_PROFILE
		const std::size_t profile_comps = num_components;
		g_boruvka_nodes_visited = 0;
		g_boruvka_dist_evals = 0;
		const auto profile_t0 = std::chrono::steady_clock::now();
#endif
		// Late rounds (few large components => mostly-pure query leaves) use the leaf-batched dual-tree;
		// early rounds use per-point. Both are exact; the choice is purely about which traversal prunes
		// better at this component count. Hard-gated to low Dim (see kBoruvkaDualMaxDim): the box-vs-box
		// bound is useless in high dimension, so above the gate we always use per-point.
		bool use_dual = false;
		if constexpr ( Dim <= kBoruvkaDualMaxDim ) {
			use_dual = ( num_components <= dual_threshold );
		}

		// Parallel over workers: each owns a contiguous block (query points in the per-point path, query
		// leaves in the dual path) and its OWN per-component bests, merged afterwards.
		detail::parallel_for( n_threads, n_workers, [&]( std::size_t w ) {
			auto& cbw = wbest_w[w];
			auto& cbu = wbest_u[w];
			auto& cbv = wbest_v[w];
			std::fill( cbw.begin(), cbw.end(), INF );  // reset this worker's bests for the round
			if ( use_dual ) {
				const auto& leaves = tree.leaves();
				const auto& order = tree.order();
				const std::size_t n_leaves = leaves.size();
				const std::size_t lblock = ( n_leaves + n_workers - 1 ) / n_workers;
				const std::size_t lo = w * lblock;
				const std::size_t hi = std::min( n_leaves, lo + lblock );
				for ( std::size_t li = lo; li < hi; ++li ) {
					const int qleaf = leaves[li];
					const auto& qn = tree.nodes()[static_cast<std::size_t>( qleaf )];
					if ( qn.comp != BoruvkaKdTree<T, Dim>::MIXED ) {
						const std::size_t cq = qn.comp;  // pure leaf: one batched descent for the whole leaf
						boruvka_dual_search( tree, tree.root(), qleaf, points, core, cq, point_comp,
											 cbw[cq], cbu[cq], cbv[cq] );
					} else {
						for ( std::size_t i = qn.start; i < qn.start + qn.count; ++i ) {
							const std::size_t q = order[i];  // mixed boundary leaf: fall back to per-point
							const std::size_t c = point_comp[q];
							boruvka_search( tree, tree.root(), points, core, q, c, point_comp, cbw[c], cbu[c], cbv[c] );
						}
					}
				}
				return;
			}
			const std::size_t lo = w * pblock;
			const std::size_t hi = std::min( n, lo + pblock );
			for ( std::size_t q = lo; q < hi; ++q ) {
				const std::size_t c = point_comp[q];
				boruvka_search( tree, tree.root(), points, core, q, c, point_comp, cbw[c], cbu[c], cbv[c] );
			}
		} );

#ifdef OPTICS_BORUVKA_PROFILE
		{
			const auto profile_t1 = std::chrono::steady_clock::now();
			const double ms = std::chrono::duration<double, std::milli>( profile_t1 - profile_t0 ).count();
			const double evals_per_pt = n > 0 ? static_cast<double>( g_boruvka_dist_evals ) / static_cast<double>( n ) : 0.0;
			std::fprintf( stderr, "[boruvka] round=%zu comps=%zu search_ms=%.3f dist_evals=%zu evals_per_pt=%.1f nodes=%zu\n",
						  profile_round, profile_comps, ms, g_boruvka_dist_evals, evals_per_pt, g_boruvka_nodes_visited );
			++profile_round;
		}
#endif
		// Merge the workers' per-component bests (min weight; ties resolved by worker order).
		std::fill( comp_best_w.begin(), comp_best_w.end(), INF );
		for ( unsigned w = 0; w < n_workers; ++w ) {
			const auto& cbw = wbest_w[w];
			for ( std::size_t c = 0; c < n; ++c ) {
				if ( cbw[c] < comp_best_w[c] ) { comp_best_w[c] = cbw[c]; comp_best_u[c] = wbest_u[w][c]; comp_best_v[c] = wbest_v[w][c]; }
			}
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
