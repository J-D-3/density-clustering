// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"

#include "../include/optics/optics.hpp"
#include "../include/optics/testdata.hpp"
#include "../include/optics/io.hpp"
#include "../include/optics/detail/hadamard.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <vector>


// Returns a sorted copy of a container (replaces the former fplus::sort usage).
template <class C>
C sorted( C c ) {
	std::sort( c.begin(), c.end() );
	return c;
}


TEST_CASE("clustering_test_1") {
	static const int N = 2;
	typedef std::array<double, N> point; //A list of N cartesian coordinates makes a point

	std::vector<point> points; //Your list of points goes here
	points = { {100,100}, {102,100}, {101,101},           //cluster 1
			   {-1,0}, {1,0}, {0,1},                     //cluster 2
			   {-100,-100}, {-102,-100}, {-101,-101}     //cluster 3
	};
	auto reach_dists = optics::compute_reachability_dists( points, 2, 10 );
	/*for( const auto& x : reach_dists){
		std::cout << x.to_string() << "; ";
	}*/

	auto clusters = optics::get_cluster_indices(reach_dists, 10);
	CHECK(clusters.size() == 3);
	CHECK( ( sorted( clusters[0] ) == std::vector <std::size_t>({ 0,1,2 }) ) );
	CHECK( ( sorted( clusters[1] ) == std::vector <std::size_t>({ 3,4,5 }) ) );
	CHECK( ( sorted( clusters[2] ) == std::vector <std::size_t>({ 6,7,8 }) ) );
}


TEST_CASE("clustering_test_2") {
	static const int N = 2;
	typedef std::array<double, N> point; //A list of N cartesian coordinates makes a point

	std::vector<point> points; //Your list of points goes here
	points = { { 100,100 },{ 102,100 },{ 101,101 },           //cluster 1
	{ -1,0 },{ 1,0 },{ 0,1 },                     //cluster 2
	{ -100,-100 },{ -102,-100 },{ -101,-101 }     //cluster 3
	};
	auto reach_dists = optics::compute_reachability_dists( points, 2 );
	/*for ( const auto& x : reach_dists ) {
		std::cout << x.to_string() << "; ";
	}*/

	auto clusters = optics::get_cluster_indices( reach_dists, 2 );
	CHECK( clusters.size() == 3 );
	CHECK( (sorted( clusters[0] ) == std::vector <std::size_t>( { 0,1,2 } )) );
	CHECK( (sorted( clusters[1] ) == std::vector <std::size_t>( { 3,4,5 } )) );
	CHECK( (sorted( clusters[2] ) == std::vector <std::size_t>( { 6,7,8 } )) );
}


TEST_CASE("clustering_test_3") {
	static const int N = 2;
	typedef std::array<double, N> point; //A list of N cartesian coordinates makes a point

	std::vector<point> points; //Your list of points goes here
	points = { {100,100}, {102,100}, {101,101},           //cluster 1
			   {1,1}, {1,0}, {0,1},                     //cluster 2
			   {50,40}, {52,40}, {51,41}     //cluster 3
	};

	auto reach_dists = optics::compute_reachability_dists( points, 2, 10 );

	auto clusters = optics::get_cluster_indices( reach_dists, 10 );
	CHECK( clusters.size() == 3 );
	CHECK( (sorted( clusters[0] ) == std::vector <std::size_t>( { 0,1,2 } )) );
	CHECK( (sorted( clusters[1] ) == std::vector <std::size_t>( { 3,4,5 } )) );
	CHECK( (sorted( clusters[2] ) == std::vector <std::size_t>( { 6,7,8 } )) );

	auto cluster_points = optics::get_cluster_points(reach_dists, 10, points);
	CHECK( cluster_points.size() == 3 );
}


TEST_CASE("epsilon_estimation_test_1") {
	static const int N = 2;
	typedef std::array<double, N> point; //A list of N cartesian coordinates makes a point

	std::vector<point> points; //Your list of points goes here
	points = { { 0,0 },{ 1,0 },{ 0,1 },
	{ 10,0 },{ 0,10 },{ 6,6 },{ 4,4 },
	{ 10,10 },{ 9,10 },{ 10,9 }
	};

	double epsilon_est = optics::epsilon_estimation( points, 3 );
	CHECK( (epsilon_est < 3.090196 && epsilon_est > 3.09019) );
}
TEST_CASE("epsilon_estimation_test_2") {
	static const int N = 3;
	typedef std::array<double, N> point; //A list of N cartesian coordinates makes a point

	std::vector<point> points; //Your list of points goes here
	points = { { 0,0,0 },{ 1,0,0 },{ 0,0,1 },{ 0,1,0 },
	{ 5,0,0 },{ 0,5,0 },{ 0,0,5 },{ 5,5,5 }
	};

	double epsilon_est = optics::epsilon_estimation( points, 3 );
	CHECK( (epsilon_est > 2.236750 && epsilon_est < 2.236751) );
}


TEST_CASE("chi_test_1") {
	std::vector<optics::reachability_dist> reach_dists = {
		{1,10.0}, {2,9.0}, {3,9.0}, {4, 5.0},//SDA
		{5,5.49}, {6,5.0},//Cluster1
		{7, 6.5},//SUA
		{8,3.0},//SDA
		{9, 2.9}, {10, 2.8},//Cluster2
		{11, 10.0},{12, 12.0}//SUA
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
   auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
   std::vector<std::pair<std::size_t, std::size_t>> exp ({ {2, 5}, {0, 11}, { 6,10 } });
   CHECK( clusters == exp );
}
TEST_CASE("chi_test_2") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,10.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 6.5 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 10.0 },{ 12, 12.0 },//SUA
		{13, 4.0},//SDA
		{14, 4.1}, {15,4.0},{ 16,3.9 },//Cluster3
		{17,5.0}//SUA
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 2, 5 },{ 0, 10 },{ 6,10 }, {11,16} } )) );
}
TEST_CASE("chi_test_3") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,11.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 6.5 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 10.0 },{ 12, 10.0 },//SUA
		{ 13, 4.0 },//SDA
		{ 14, 4.1 },{ 15,4.0 },{ 16,3.9 },//Cluster3
		{ 17,12.0 }//SUA
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 2, 5 },{ 0, 9 },{ 6,10 },{0,16},{ 11,16 } } )) );
}
TEST_CASE("chi_test_4") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,12.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 6.5 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 10.0 },{ 12, 10.0 },//SUA
		{ 13, 4.0 },//SDA
		{ 14, 4.1 },{ 15,4.0 },{ 16,3.9 },//Cluster3
		{ 17,11.0 }//SUA
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 2, 5 },{ 0, 9 },{ 6,10 },{0,16},{ 11,16 } } )) );
}
TEST_CASE("chi_test_5") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,12.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 6.5 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 10.0 },{ 12, 10.0 },//SUA
		{ 13, 4.0 },//SDA
		{ 14, 4.1 },{ 15,4.0 },{ 16,3.9 },//Cluster3
		{ 17,12.0 }//SUA
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 2, 5 },{ 0, 9 },{ 6,10 },{0,16},{ 11,16 } } )) );
}
TEST_CASE("chi_test_6") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,12.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 6.5 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 10.0 },{ 12, 10.0 },//SUA
		{ 13, 4.0 },//SDA
		{ 14, 4.1 },{ 15,4.0 },{ 16,3.9 },//Cluster3
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 2, 5 },{ 0, 9 },{ 6,10 },{2,15},{ 11,15 } } )) );
}
TEST_CASE("chi_test_7") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,12.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 11.0 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 9.89 },{ 12, 9.89 },//SUA
		{ 13, 4.0 },//SDA
		{ 14, 4.1 },{ 15,4.0 },{ 16,3.9 },//Cluster3
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 0, 5 },{ 6, 9 },{6,15},{ 11,15 } } )) );
}
TEST_CASE("chi_test_8") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,12.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 11.0 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 9.89 },{ 12, 9.91 },//SUA
		{ 13, 4.0 },//SDA
		{ 14, 4.1 },{ 15,4.0 },{ 16,3.9 },//Cluster3
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 0, 5 },{ 6, 9 },{ 11,15 } } )) );
}
TEST_CASE("chi_test_9") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 0, 5.0 }, { 1,5.49 },{ 2,5.0 },//Cluster1
		{ 3, 11.0 },//SUA
		{ 4,3.0 },//SDA
		{ 5, 2.9 },{ 6, 2.8 },//Cluster2
		{ 7, 9.89 },{ 8, 9.9 },//SUA
		{ 9, 4.0 },//SDA
		{ 10, 4.1 },{ 11,4.0 },{ 12,3.9 },//Cluster3
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 0, 2 },{ 3, 6 },{ 3,12 }, {8,12} } )) );
}
TEST_CASE("chi_test_10") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 0, 5.0 }, { 1,5.49 },{ 2,5.0 },//Cluster1
		{ 3, 11.0 },//SUA
		{ 4,3.0 },//SDA
		{ 5, 2.9 },{ 6, 2.8 },//Cluster2
		{ 7, 9.89 },{ 8, 9.91 },//SUA
		{ 9, 4.0 },//SDA
		{ 10, 4.1 },{ 11,4.0 },{ 12,3.9 },//Cluster3
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts );
	CHECK( (clusters == std::vector<std::pair<std::size_t, std::size_t>>( { { 0, 2 },{ 3, 6 }, {8,12} } )) );
}


// The neighbor sets are identical regardless of how they are acquired, so the
// reachability ordering must be byte-for-byte identical across Precompute (any
// thread count) and OnDemand. This pins the P4 parallelism/mode machinery.
TEST_CASE("neighbor_mode_tests") {
	static const int N = 2;
	typedef std::array<double, N> point;

	std::mt19937 gen( 12345 );
	std::normal_distribution<double> jitter( 0.0, 1.5 );
	const std::array<point, 3> centers = { point{ 0, 0 }, point{ 40, 5 }, point{ 10, 35 } };

	std::vector<point> points;
	points.reserve( 600 );
	for ( const auto& c : centers ) {
		for ( int i = 0; i < 200; ++i ) {
			points.push_back( { c[0] + jitter( gen ), c[1] + jitter( gen ) } );
		}
	}

	const std::size_t min_pts = 8;
	const auto baseline = optics::compute_reachability_dists( points, min_pts, -1.0, optics::NeighborMode::Precompute, 1 );

	const auto precompute_4t = optics::compute_reachability_dists( points, min_pts, -1.0, optics::NeighborMode::Precompute, 4 );
	const auto on_demand = optics::compute_reachability_dists( points, min_pts, -1.0, optics::NeighborMode::OnDemand, 1 );

	CHECK( baseline.size() == points.size() );
	CHECK( precompute_4t == baseline );
	CHECK( on_demand == baseline );

	std::cout << "Neighbor-mode parity tests successful!" << std::endl;
}


TEST_CASE("core_dist_mode parity: Knn matches Scan") {
	// The k-NN core-distance path (issue #24) must produce results bit-identical to
	// the eps-neighborhood scan: it returns the same kth-nearest distance value.
	using point = std::array<double, 2>;

	// Gaussian blobs + uniform noise + a dense block of identical points to exercise
	// the dense-neighborhood regime Knn targets (and tie handling at the kth distance).
	auto points = optics::testdata::make_blobs<double, 2>( 4, 150, 50.0, 2.0, 123 );
	const auto noise = optics::testdata::uniform_noise<double, 2>( 100, -60.0, 60.0, 9 );
	points.insert( points.end(), noise.begin(), noise.end() );
	for ( int i = 0; i < 40; ++i ) { points.push_back( point{ 5.0, 5.0 } ); }

	const std::size_t min_pts = 8;
	const auto scan = optics::compute_reachability_dists(
		points, min_pts, -1.0, optics::NeighborMode::Precompute, 1, optics::CoreDistMode::Scan );
	const auto knn = optics::compute_reachability_dists(
		points, min_pts, -1.0, optics::NeighborMode::Precompute, 1, optics::CoreDistMode::Knn );
	const auto knn_ondemand = optics::compute_reachability_dists(
		points, min_pts, -1.0, optics::NeighborMode::OnDemand, 1, optics::CoreDistMode::Knn );

	CHECK( scan.size() == points.size() );
	CHECK( ( knn == scan ) );
	CHECK( ( knn_ondemand == scan ) );

	std::cout << "CoreDistMode Knn/Scan parity tests successful!" << std::endl;
}


TEST_CASE("approximate backend: bounded recall + interchangeable clustering") {
	// The approximate backend (issue #28) trades boundary recall for speed via
	// nanoflann's eps-approximation. On well-separated blobs in a higher-D space it
	// should keep high neighbor recall and recover the same dense clusters as exact.
	static const int N = 8;
	using point = std::array<double, N>;

	// The alias must be a genuinely approximate configuration, and the default exact;
	// otherwise this test would silently compare a backend against itself.
	static_assert( optics::ApproxNanoflannBackend<double, N>::search_eps > 0.0f, "approx backend must use eps>0" );
	static_assert( optics::NanoflannBackend<double, N>::search_eps == 0.0f, "default backend must be exact" );

	const auto points = optics::testdata::make_blobs<double, N>( 5, 200, 60.0, 2.0, 77 );
	const double eps = 12.0;
	const std::size_t min_pts = 10;

	const optics::NanoflannBackend<double, N> exact( points );
	const optics::ApproxNanoflannBackend<double, N> approx( points );  // eps = 0.1 (lossless here)

	// Neighbor recall: the approximate set is a subset of exact; measure coverage.
	std::size_t exact_total = 0, found_total = 0;
	for ( std::size_t i = 0; i < points.size(); i += 5 ) {
		std::vector<std::size_t> a, b;
		exact.radius_search( points[i], eps, a );
		approx.radius_search( points[i], eps, b );
		const std::set<std::size_t> bs( b.begin(), b.end() );
		exact_total += a.size();
		for ( const auto idx : a ) { if ( bs.count( idx ) ) { ++found_total; } }
	}
	const double recall = static_cast<double>( found_total ) / static_cast<double>( exact_total );
	CHECK( recall > 0.85 );

	// End-to-end: both backends recover the 5 dense blobs under a threshold cut.
	const auto reach_exact = optics::compute_reachability_dists<double, N>( points, min_pts, eps );
	const auto reach_approx = optics::compute_reachability_dists<double, N, optics::ApproxNanoflannBackend<double, N>>( points, min_pts, eps );
	const auto count_large = []( const std::vector<std::vector<std::size_t>>& cls ) {
		std::size_t k = 0;
		for ( const auto& c : cls ) { if ( c.size() >= 100 ) { ++k; } }
		return k;
	};
	CHECK( count_large( optics::get_cluster_indices( reach_exact, eps ) ) == 5 );
	CHECK( count_large( optics::get_cluster_indices( reach_approx, eps ) ) == 5 );

	std::cout << "Approximate backend recall = " << recall << std::endl;
}


TEST_CASE("chi_test_11") {
   std::vector<optics::reachability_dist> reach_dists = {
	   {0,-1.000000}, {1,-1.000000}, {2,-1.000000}, {3,-1.000000}, {4,-1.000000}, {5,-1.000000}, {6,-1.000000}, {7,-1.000000}, {8,-1.000000},
	   {9,-1.000000}, {10,-1.000000}, {11,-1.000000}, {12,-1.000000}, {13,-1.000000}, {14,-1.000000}, {15,-1.000000}, {16,-1.000000},
	   {17,-1.000000}, {18,-1.000000}, {19,-1.000000}, {20,-1.000000}, {21,-1.000000}, {22,-1.000000}, {23,-1.000000}, {24,-1.000000},
	   {25,-1.000000}, {26,-1.000000}, {27,-1.000000}, {28,-1.000000}, {29,-1.000000}, {30,-1.000000}, {31,-1.000000}, {32,-1.000000},
	   {33,-1.000000}, {34,-1.000000}, {35,-1.000000}, {36,-1.000000}, {37,-1.000000}, {38,-1.000000}, {39,-1.000000}, {40,-1.000000},
	   {41,-1.000000}, {42,-1.000000}, {43,-1.000000}, {44,-1.000000}, {45,-1.000000}, {46,-1.000000}, {47,-1.000000}, {48,-1.000000},
	   {49,-1.000000}, {50,-1.000000}, {51,-1.000000}, {52,-1.000000}, {53,-1.000000}, {54,-1.000000}, {55,-1.000000}, {56,-1.000000},
	   {57,-1.000000}, {58,-1.000000}, {59,-1.000000}, {60,-1.000000}, {61,-1.000000}, {62,-1.000000}, {63,-1.000000}, {64,-1.000000},
	   {65,-1.000000}, {66,-1.000000}, {67,-1.000000}, {68,-1.000000}, {69,-1.000000}, {70,-1.000000}, {71,-1.000000}, {72,-1.000000},
	   {73,-1.000000}, {74,-1.000000}, {75,-1.000000}, {76,-1.000000}, {77,-1.000000}, {78,-1.000000}, {79,-1.000000}, {80,-1.000000},
	   {81,-1.000000}, {82,-1.000000}, {83,-1.000000}, {84,-1.000000}, {85,-1.000000}, {86,-1.000000}, {87,-1.000000}, {88,-1.000000},
	   {89,-1.000000}, {90,-1.000000}, {91,-1.000000}, {92,-1.000000}, {93,-1.000000}, {94,-1.000000}, {95,-1.000000}, {96,-1.000000},
	   {97,-1.000000}, {98,-1.000000}, {99,-1.000000}, {100,-1.000000}, {101,-1.000000}, {102,-1.000000}, {103,-1.000000}, {104,-1.000000},
	   {105,-1.000000}, {106,-1.000000}, {107,-1.000000}, {108,-1.000000}, {109,-1.000000}, {110,-1.000000}, {111,-1.000000}, {112,-1.000000},
	   {113,-1.000000}, {114,-1.000000}, {115,-1.000000}, {116,-1.000000}, {117,-1.000000}, {118,-1.000000}, {119,-1.000000}, {120,-1.000000},
	   {121,-1.000000}, {122,-1.000000}, {123,-1.000000}, {124,-1.000000}, {125,-1.000000}, {126,-1.000000}, {127,-1.000000}, {128,-1.000000},
	   {129,-1.000000}, {130,-1.000000}, {131,-1.000000}, {132,-1.000000}, {133,-1.000000}, {134,-1.000000}, {135,-1.000000}, {136,-1.000000},
	   {137,-1.000000}, {138,-1.000000}, {139,-1.000000}, {140,-1.000000}, {141,-1.000000}, {142,-1.000000}, {143,-1.000000}, {144,-1.000000},
	   {145,-1.000000}, {146,-1.000000}, {147,-1.000000}, {148,-1.000000}, {149,-1.000000}, {150,-1.000000}, {164,9.433981}, {183,8.944272},
	   {184,8.944272}, {185,8.544004}, {201,8.544004}, {193,7.000000}, {194,7.000000}, {210,7.000000}, {229,7.000000}, {247,7.000000},
	   {274,8.062258}, {275,8.062258}, {248,8.246211}, {211,7.280110}, {230,7.280110}, {243,7.280110}, {276,7.280110}, {277,7.071068},
	   {299,7.071068}, {326,7.071068}, {327,7.000000}, {371,7.000000}, {388,7.000000}, {389,7.000000}, {397,7.211103}, {361,7.280110},
	   {408,7.280110}, {212,8.062258}, {328,8.062258}, {345,8.062258}, {262,7.810250}, {202,7.810250}, {221,7.810250}, {231,7.810250},
	   {278,7.810250}, {329,7.810250}, {330,7.810250}, {346,7.810250}, {372,7.810250}, {414,7.810250}, {415,7.280110}, {429,7.280110},
	   {430,7.280110}, {431,7.280110}, {443,7.280110}, {454,7.280110}, {468,7.280110}, {486,7.280110}, {523,7.615773}, {347,8.062258},
	   {479,8.062258}, {313,8.246211}, {381,8.246211}, {312,7.000000}, {360,7.000000}, {421,7.000000}, {427,7.000000}, {428,6.324555},
	   {441,5.830952}, {442,5.099020}, {464,5.099020}, {465,5.000000}, {466,4.242641}, {467,4.242641}, {478,4.242641}, {501,4.242641},
	   {485,4.472136}, {500,6.082763}, {522,6.082763}, {552,6.082763}, {538,6.403124}, {539,6.403124}, {540,6.403124}, {547,6.403124},
	   {463,6.708204}, {541,7.000000}, {542,7.000000}, {563,7.000000}, {573,7.000000}, {311,7.071068}, {606,7.071068}, {584,7.211103},
	   {623,7.211103}, {537,7.810250}, {644,8.062258}, {413,8.246211}, {546,8.246211}, {662,8.485281}, {412,8.602325}, {344,8.602325},
	   {273,8.062258}, {325,8.062258}, {237,8.062258}, {356,8.062258}, {357,8.062258}, {358,7.810250}, {359,7.071068}, {396,7.071068},
	   {407,7.071068}, {380,7.810250}, {406,7.810250}, {236,8.246211}, {499,8.602325}, {524,8.944272}, {228,9.000000}, {331,9.000000},
	   {382,9.000000}, {390,9.000000}, {444,9.000000}, {632,9.055385}, {694,9.055385}, {709,9.055385}, {726,9.055385}, {157,9.219544},
	   {165,9.219544}, {741,9.219544}, {195,9.433981}, {607,9.433981}, {614,9.433981}, {653,9.433981}, {633,7.280110}, {585,5.099020},
	   {608,5.099020}, {634,5.099020}, {645,5.000000}, {654,4.123106}, {663,3.605551}, {664,3.605551}, {665,3.000000}, {679,3.000000},
	   {696,3.000000}, {697,3.000000}, {711,3.000000}, {712,3.000000}, {727,3.000000}, {713,3.162278}, {680,3.162278}, {646,3.162278},
	   {666,3.162278}, {667,3.162278}, {647,3.000000}, {668,3.000000}, {681,3.000000}, {682,3.000000}, {698,3.000000}, {699,2.236068},
	   {714,2.236068}, {715,2.236068}, {729,2.236068}, {700,2.828427}, {683,2.236068}, {701,2.236068}, {730,2.236068}, {731,2.236068},
	   {669,2.828427}, {728,2.828427}, {744,2.828427}, {745,2.828427}, {750,2.828427}, {751,2.828427}, {732,3.000000}, {733,3.000000},
	   {746,3.000000}, {760,3.000000}, {635,3.162278}, {749,3.162278}, {759,3.162278}, {765,3.162278}, {770,3.162278}, {771,3.162278},
	   {784,3.162278}, {624,3.605551}, {625,3.605551}, {626,3.605551}, {655,3.605551}, {670,3.605551}, {678,3.605551}, {752,3.605551},
	   {753,3.605551}, {764,3.605551}, {766,3.605551}, {772,3.605551}, {783,3.605551}, {797,3.605551}, {798,3.605551}, {684,4.000000},
	   {598,4.123106}, {648,4.123106}, {685,4.123106}, {716,4.123106}, {773,4.123106}, {799,4.123106}, {627,4.242641}, {734,4.242641},
	   {774,4.242641}, {796,4.242641}, {586,4.472136}, {597,4.472136}, {695,4.472136}, {785,4.472136}, {677,5.000000}, {782,5.000000},
	   {800,5.000000}, {827,5.385165}, {833,5.385165}, {649,6.708204}, {819,6.708204}, {843,6.708204}, {628,7.000000}, {710,7.000000},
	   {742,7.000000}, {743,7.000000}, {786,7.000000}, {820,7.000000}, {848,7.000000}, {636,7.071068}, {847,7.071068}, {862,7.071068},
	   {587,7.211103}, {656,7.211103}, {525,7.810250}, {686,8.246211}, {526,8.485281}, {832,8.485281}, {884,8.544004}, {516,8.602325},
	   {553,8.602325}, {574,8.602325}, {503,8.485281}, {422,8.000000}, {445,8.000000}, {470,8.000000}, {480,8.000000}, {510,8.000000},
	   {527,7.810250}, {528,7.810250}, {529,7.810250}, {554,7.810250}, {588,7.810250}, {409,8.246211}, {398,9.000000}, {432,9.000000},
	   {487,9.000000}, {455,9.219544}, {469,9.219544}, {502,9.219544}, {504,9.219544}, {899,9.219544}, {900,9.219544}, {405,9.899495},
	   {440,9.899495}, {484,9.899495}, {740,9.899495}, {795,9.899495}, {813,9.899495}, {818,9.899495}, {891,9.899495}, {151,-1.000000},
	   {152,-1.000000}, {153,-1.000000}, {154,-1.000000}, {155,-1.000000}, {173,9.433981}, {180,9.433981}, {191,9.433981}, {172,9.000000},
	   {171,9.000000}, {161,9.000000}, {190,9.000000}, {198,8.062258}, {217,8.062258}, {227,7.810250}, {234,6.708204}, {246,5.830952},
	   {256,5.830952}, {241,3.605551}, {242,3.605551}, {245,3.605551}, {257,3.605551}, {271,3.605551}, {281,3.605551}, {270,3.162278},
	   {282,3.162278}, {283,3.162278}, {284,2.828427}, {294,2.828427}, {295,2.828427}, {296,2.828427}, {304,2.828427}, {305,2.828427},
	   {306,2.828427}, {297,3.000000}, {303,3.000000}, {351,3.000000}, {293,3.162278}, {291,3.162278}, {253,3.162278}, {254,3.162278},
	   {255,3.162278}, {268,3.162278}, {269,3.162278}, {292,3.162278}, {302,3.162278}, {307,3.162278}, {319,3.162278}, {320,3.162278},
	   {321,3.162278}, {336,3.162278}, {337,3.162278}, {338,3.162278}, {349,3.162278}, {350,3.162278}, {352,3.162278}, {353,3.162278},
	   {365,3.162278}, {366,3.162278}, {367,3.162278}, {375,3.162278}, {376,3.162278}, {377,3.162278}, {393,3.605551}, {394,3.605551},
	   {402,3.605551}, {392,4.000000}, {410,4.000000}, {411,4.000000}, {226,4.123106}, {258,4.123106}, {374,4.123106}, {425,4.123106},
	   {322,4.242641}, {339,4.242641}, {385,4.242641}, {340,4.472136}, {395,4.472136}, {401,4.472136}, {403,4.472136}, {207,5.000000},
	   {259,5.000000}, {290,5.000000}, {364,5.385165}, {458,5.830952}, {418,6.324555}, {459,6.708204}, {451,5.830952}, {474,5.830952},
	   {452,5.830952}, {426,5.830952}, {453,5.830952}, {460,5.830952}, {482,5.830952}, {483,5.830952}, {507,5.656854}, {508,5.656854},
	   {519,5.656854}, {492,5.830952}, {493,5.830952}, {494,5.830952}, {495,5.830952}, {496,5.830952}, {550,5.830952}, {497,6.082763},
	   {532,6.082763}, {513,6.324555}, {520,6.324555}, {560,6.324555}, {568,6.708204}, {579,6.708204}, {595,6.708204}, {601,6.708204},
	   {386,7.000000}, {179,7.071068}, {189,7.071068}, {378,7.071068}, {384,7.071068}, {545,7.071068}, {602,7.071068}, {170,7.211103},
	   {419,7.211103}, {603,7.211103}, {613,7.211103}, {619,7.211103}, {630,7.211103}, {676,7.211103}, {160,7.280110}, {472,7.280110},
	   {450,7.071068}, {400,7.071068}, {435,7.071068}, {457,7.071068}, {471,7.071068}, {473,7.071068}, {481,7.071068}, {491,7.071068},
	   {506,7.071068}, {511,6.403124}, {518,6.403124}, {531,6.403124}, {549,7.071068}, {559,7.071068}, {578,7.071068}, {566,7.211103},
	   {591,7.211103}, {592,7.071068}, {610,7.071068}, {611,7.071068}, {617,7.071068}, {639,7.071068}, {640,7.071068}, {593,7.211103},
	   {512,7.280110}, {580,7.280110}, {272,7.615773}, {594,7.615773}, {218,7.810250}, {672,8.000000}, {159,8.062258}, {354,8.062258},
	   {260,7.071068}, {298,7.071068}, {308,7.071068}, {309,7.071068}, {323,7.071068}, {341,7.071068}, {368,7.071068}, {369,7.071068},
	   {387,7.071068}, {379,7.211103}, {436,7.211103}, {199,7.810250}, {208,7.810250}, {209,7.810250}, {219,7.810250}, {533,8.062258},
	   {558,8.062258}, {565,8.062258}, {581,8.062258}, {582,8.062258}, {692,8.062258}, {693,8.062258}, {707,7.615773}, {721,7.615773},
	   {755,7.615773}, {722,7.280110}, {737,7.280110}, {756,7.280110}, {757,7.071068}, {758,7.071068}, {708,6.324555}, {738,6.324555},
	   {762,6.324555}, {768,6.324555}, {778,6.324555}, {793,6.324555}, {816,6.324555}, {817,6.324555}, {767,7.280110}, {779,7.615773},
	   {812,7.615773}, {830,7.615773}, {841,7.615773}, {846,7.615773}, {852,7.615773}, {704,8.062258}, {777,8.062258}, {801,8.062258},
	   {825,8.062258}, {567,8.246211}, {612,8.246211}, {642,8.246211}, {652,8.246211}, {792,8.246211}, {811,8.246211}, {790,7.280110},
	   {791,7.211103}, {809,7.211103}, {810,5.656854}, {823,5.656854}, {808,5.656854}, {824,5.656854}, {829,4.472136}, {838,4.472136},
	   {839,4.472136}, {845,4.472136}, {849,4.472136}, {850,5.000000}, {828,5.656854}, {837,5.656854}, {855,5.830952}, {856,6.000000},
	   {878,6.000000}, {851,6.324555}, {788,7.071068}, {789,7.071068}, {844,7.071068}, {868,7.071068}, {903,7.071068}, {904,7.071068},
	   {840,7.211103}, {888,7.211103}, {867,8.062258}, {877,8.062258}, {897,8.062258}, {706,8.246211}, {872,8.246211}, {873,8.246211},
	   {498,8.485281}, {866,8.544004}, {705,8.602325}, {719,8.602325}, {561,8.944272}, {569,8.944272}, {618,8.944272}, {651,8.944272},
	   {660,8.944272}, {673,8.944272}, {674,8.944272}, {675,8.944272}, {691,8.944272}, {720,8.944272}, {916,8.944272}, {641,9.055385},
	   {913,9.055385}, {490,9.219544}, {643,9.219544}, {739,9.219544}, {889,9.219544}, {893,9.219544}, {894,9.219544}, {917,9.219544},
	   {562,9.848858}, {505,9.899495}, {736,9.899495}, {928,9.899495}, {156,-1.000000}, {158,-1.000000}, {162,-1.000000}, {163,-1.000000},
	   {166,-1.000000}, {167,-1.000000}, {168,-1.000000}, {169,-1.000000}, {174,-1.000000}, {175,-1.000000}, {176,-1.000000}, {177,-1.000000},
	   {178,-1.000000}, {181,-1.000000}, {182,-1.000000}, {186,-1.000000}, {196,9.433981}, {203,9.433981}, {204,9.433981}, {222,9.433981},
	   {223,9.433981}, {224,8.246211}, {238,8.246211}, {263,8.246211}, {286,8.246211}, {301,8.246211}, {300,8.544004}, {187,-1.000000},
	   {188,-1.000000}, {192,-1.000000}, {197,-1.000000}, {200,-1.000000}, {205,-1.000000}, {206,-1.000000}, {213,-1.000000}, {214,-1.000000},
	   {215,-1.000000}, {216,-1.000000}, {220,-1.000000}, {225,-1.000000}, {232,-1.000000}, {233,-1.000000}, {235,-1.000000}, {239,-1.000000},
	   {240,-1.000000}, {244,-1.000000}, {249,-1.000000}, {250,-1.000000}, {251,-1.000000}, {252,-1.000000}, {261,-1.000000}, {264,-1.000000},
	   {265,-1.000000}, {266,-1.000000}, {267,-1.000000}, {279,-1.000000}, {280,-1.000000}, {285,-1.000000}, {287,-1.000000}, {288,-1.000000},
	   {289,-1.000000}, {310,-1.000000}, {314,-1.000000}, {315,-1.000000}, {316,-1.000000}, {317,-1.000000}, {318,-1.000000}, {324,-1.000000},
	   {332,-1.000000}, {333,-1.000000}, {334,-1.000000}, {335,-1.000000}, {342,-1.000000}, {343,-1.000000}, {348,-1.000000}, {355,-1.000000},
	   {362,-1.000000}, {363,-1.000000}, {370,-1.000000}, {373,-1.000000}, {383,-1.000000}, {391,-1.000000}, {399,-1.000000}, {404,-1.000000},
	   {416,-1.000000}, {417,-1.000000}, {420,-1.000000}, {423,-1.000000}, {424,-1.000000}, {433,-1.000000}, {434,-1.000000}, {437,-1.000000},
	   {438,-1.000000}, {439,-1.000000}, {446,-1.000000}, {447,-1.000000}, {448,-1.000000}, {449,-1.000000}, {456,-1.000000}, {461,-1.000000},
	   {462,-1.000000}, {475,-1.000000}, {476,-1.000000}, {477,-1.000000}, {488,-1.000000}, {489,-1.000000}, {509,-1.000000}, {514,-1.000000},
	   {515,-1.000000}, {517,-1.000000}, {521,-1.000000}, {530,-1.000000}, {534,-1.000000}, {535,-1.000000}, {536,-1.000000}, {543,-1.000000},
	   {544,-1.000000}, {548,-1.000000}, {551,-1.000000}, {555,-1.000000}, {556,-1.000000}, {557,-1.000000}, {564,-1.000000}, {570,-1.000000},
	   {571,-1.000000}, {572,-1.000000}, {575,-1.000000}, {576,-1.000000}, {577,-1.000000}, {583,-1.000000}, {589,-1.000000}, {590,-1.000000},
	   {596,-1.000000}, {599,-1.000000}, {600,-1.000000}, {604,-1.000000}, {605,-1.000000}, {609,-1.000000}, {615,-1.000000}, {616,-1.000000},
	   {620,-1.000000}, {621,-1.000000}, {622,-1.000000}, {629,-1.000000}, {631,-1.000000}, {637,-1.000000}, {638,-1.000000}, {650,-1.000000},
	   {657,-1.000000}, {658,-1.000000}, {659,-1.000000}, {661,-1.000000}, {671,-1.000000}, {687,-1.000000}, {688,-1.000000}, {689,-1.000000},
	   {690,-1.000000}, {702,-1.000000}, {703,-1.000000}, {717,-1.000000}, {718,-1.000000}, {723,-1.000000}, {724,-1.000000}, {725,-1.000000},
	   {735,-1.000000}, {747,-1.000000}, {748,-1.000000}, {754,-1.000000}, {761,-1.000000}, {763,-1.000000}
   };

   {
	   double chi = 0.02;
	   double steep_area_min_diff = 0.15;
	   std::size_t min_pts = 5;

	   auto clusters = optics::get_chi_clusters_flat( reach_dists, chi, min_pts, steep_area_min_diff );
	   std::vector<std::pair<std::size_t, std::size_t>> expected_result =
	   { { 155, 162 },{ 203, 225 },{ 295, 299 },{ 300, 304 },{ 271, 358 },{ 270, 372 },{ 150, 407 },{ 422, 493 },{ 590, 607 },{ 626, 642 },{ 412, 684 },{ 700, 711 } };
	   CHECK( (clusters == expected_result) );
   }

   {
	   double chi = 0.1;
	   double steep_area_min_diff = 0.02;
	   std::size_t min_pts = 8;

	   auto clusters2 = optics::get_chi_clusters_flat( reach_dists, chi, min_pts, steep_area_min_diff );
	   std::vector<std::pair<std::size_t, std::size_t>> expected_result =
		{ {155, 160}, {208, 217}, {276, 321}, {271, 355}, {150, 407}, {425, 470},
		{425, 487}, {598, 606}, {626, 642}, {623, 650}, {412, 684}, {700, 711} };
	   CHECK( (clusters2 == expected_result) );
   }
}


TEST_CASE("tree_tests") {
	typedef std::pair<std::size_t, std::size_t> cluster;
	typedef optics::Node<cluster> Node;

	{
		optics::Node<cluster> n( { 0,5 } );
		optics::Tree<cluster> T( n );
		auto c = T.flatten();
		CHECK( c == std::vector<cluster>( { cluster( 0, 5 ) } ) );
	}

	{
		optics::Node<cluster> n( { 0,5 } );
		optics::Tree<cluster> T( n );
		auto c = T.flatten();
		CHECK( c == std::vector<cluster>( { cluster( 0, 5 ) } ) );
	}

	{
		optics::Node<cluster> n( { 0,5 } );
		optics::Tree<cluster> T( n );
		auto& root = T.get_root();
		root.add_children( std::vector<Node>( { Node( {1,1} ), Node( {1,2} ), Node( {1,3} ) } ) );
		std::size_t idx = 1;
		for ( auto& child : root.get_children() ) {
			child.add_child( Node( { 2, idx++ } ) );
		}
		auto c = T.flatten();
	}
}

bool trees_are_equal( const optics::Node<optics::chi_cluster_indices>& t1, const optics::Node<optics::chi_cluster_indices>& t2 ) {
	if ( t1.get_data() != t2.get_data() ) {
		return false;
	}
	if ( t1.get_children().size() != t2.get_children().size() ) {
		return false;
	}
	if ( t1.get_children().size() == 0 && t2.get_children().size() == 0 && t1.get_data() == t2.get_data() ) {
		return true;
	}
	const auto& c1 = t1.get_children();
	const auto& c2 = t2.get_children();
	for ( std::size_t i = 0; i < c1.size(); ++i ) {
		if ( !trees_are_equal( c1[i], c2[i] ) ) { return false; }
	}
	return true;
}

TEST_CASE("chi_cluster_tree_tests_1") {
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,10.0 },{ 2,9.0 },{ 3,9.0 },{ 4, 5.0 },//SDA
		{ 5,5.49 },{ 6,5.0 },//Cluster1
		{ 7, 6.5 },//SUA
		{ 8,3.0 },//SDA
		{ 9, 2.9 },{ 10, 2.8 },//Cluster2
		{ 11, 10.0 },{ 12, 12.0 }//SUA
	};
	double chi = 0.1;
	std::size_t min_pts = 4;
	auto clusters = optics::get_chi_clusters( reach_dists, chi, min_pts );
	CHECK( clusters.size() == 1 );

	optics::cluster_tree expected_result =
	{
		optics::cluster_tree{
			{ {0,11},
				{ { {2,5},  {} },
				  { {6,10}, {} }
				}
			}
		}
	};
	CHECK( trees_are_equal(clusters.front().get_root(), expected_result.get_root() ) );
}


TEST_CASE("chi_cluster_tree_tests_2") {
	std::vector<optics::chi_cluster_indices> flat_clusters = {
		{0,4}, {0,8}, {5,7},
		{9,10}, {12,13}, {9,17}, {11,17}, {13,14}, {8,20}
	};

	typedef optics::Node<optics::chi_cluster_indices> Node;
	std::vector<optics::cluster_tree> expected_result(
	{
		optics::cluster_tree{
			Node({ 0,8 },
			  {
				{ { 0,4 },{} },
				{ { 5,7 },{} }
			  })
			},
		optics::cluster_tree{
			Node({8,20},
			  {
				  { {9,17},
					{
						{{9,10},{}},
						{{11,17},{
							{{12,13},{}},
							{{13,14},{}}
						 }}
					}}
			  })}
	});

	auto clusters = optics::flat_clusters_to_tree( flat_clusters );
	CHECK( clusters.size() == 2 );
	CHECK( trees_are_equal( clusters[0].get_root(), expected_result[0].get_root() ) );
	CHECK( trees_are_equal( clusters[1].get_root(), expected_result[1].get_root() ) );
}


// Naive O(n^2) OPTICS reference: brute-force neighbor scan + linear-scan seed
// selection. Uses the same distance primitives so the ordering must match the
// optimized library bit-for-bit; the independence is in the control flow
// (no kd-tree, no heap), which is exactly what we want to validate.
template <class T, std::size_t Dim>
std::vector<optics::reachability_dist> brute_force_optics(
	const std::vector<std::array<T, Dim>>& points, std::size_t min_pts, double eps ) {
	const std::size_t n = points.size();
	const double eps2 = eps * eps;
	std::vector<char> processed( n, 0 );
	std::vector<double> reach( n, -1.0 );
	std::vector<std::size_t> order;
	std::set<std::size_t> seeds;

	const auto neighbors = [&]( std::size_t i ) {
		std::vector<std::size_t> r;
		for ( std::size_t j = 0; j < n; ++j ) {
			if ( optics::detail::square_dist( points[i], points[j] ) <= eps2 ) { r.push_back( j ); }
		}
		return r;
	};
	const auto core_dist = [&]( std::size_t i, const std::vector<std::size_t>& nb ) -> std::optional<double> {
		if ( nb.size() < min_pts ) { return std::nullopt; }
		std::vector<double> sq;
		for ( std::size_t j : nb ) { sq.push_back( optics::detail::square_dist( points[i], points[j] ) ); }
		std::sort( sq.begin(), sq.end() );
		return std::sqrt( sq[min_pts - 1] );
	};
	const auto relax = [&]( std::size_t i, const std::vector<std::size_t>& nb, double cd ) {
		for ( std::size_t o : nb ) {
			if ( processed[o] ) { continue; }
			const double nrd = std::max( cd, optics::detail::dist( points[i], points[o] ) );
			if ( reach[o] < 0.0 || nrd < reach[o] ) { reach[o] = nrd; seeds.insert( o ); }
		}
	};

	for ( std::size_t i = 0; i < n; ++i ) {
		if ( processed[i] ) { continue; }
		processed[i] = 1;
		order.push_back( i );
		auto nb = neighbors( i );
		if ( auto cd = core_dist( i, nb ) ) { relax( i, nb, *cd ); }
		while ( !seeds.empty() ) {
			const std::size_t best = *std::min_element( seeds.begin(), seeds.end(),
				[&]( std::size_t a, std::size_t b ) { return reach[a] != reach[b] ? reach[a] < reach[b] : a < b; } );
			seeds.erase( best );
			if ( processed[best] ) { continue; }
			processed[best] = 1;
			order.push_back( best );
			auto nb2 = neighbors( best );
			if ( auto cd = core_dist( best, nb2 ) ) { relax( best, nb2, *cd ); }
		}
	}
	std::vector<optics::reachability_dist> result;
	for ( std::size_t idx : order ) { result.emplace_back( idx, reach[idx] ); }
	return result;
}

TEST_CASE("reference: library matches brute-force O(n^2) OPTICS") {
	// 2D: three well-separated blobs (no pairwise distance near eps).
	{
		const std::vector<std::array<double, 2>> centers = { { 0, 0 }, { 40, 0 }, { 20, 35 } };
		const auto points = optics::testdata::gaussian_blobs<double, 2>( centers, 40, 1.5 );
		const std::size_t min_pts = 5;
		const double eps = 15.0;
		const auto ref = brute_force_optics( points, min_pts, eps );
		CHECK( optics::compute_reachability_dists( points, min_pts, eps, optics::NeighborMode::Precompute, 4 ) == ref );
		CHECK( optics::compute_reachability_dists( points, min_pts, eps, optics::NeighborMode::OnDemand, 1 ) == ref );
	}
	// 3D, with some noise points (undefined core-distance paths).
	{
		const std::vector<std::array<double, 3>> centers = { { 0, 0, 0 }, { 50, 0, 0 }, { 0, 50, 0 } };
		auto points = optics::testdata::gaussian_blobs<double, 3>( centers, 50, 2.0 );
		const auto noise = optics::testdata::uniform_noise<double, 3>( 15, -30.0, 80.0, 99 );
		points.insert( points.end(), noise.begin(), noise.end() );
		const std::size_t min_pts = 6;
		const double eps = 10.0;
		const auto ref = brute_force_optics( points, min_pts, eps );
		CHECK( optics::compute_reachability_dists( points, min_pts, eps ) == ref );
	}
}

TEST_CASE("edge cases and degenerate inputs") {
	using P2 = std::array<double, 2>;

	// Empty input -> empty result.
	CHECK( optics::compute_reachability_dists( std::vector<P2>{}, 5 ).empty() );

	// min_pts must be >= 1.
	std::vector<P2> two = { { 0, 0 }, { 1, 1 } };
	CHECK_THROWS_AS( optics::compute_reachability_dists( two, 0 ), std::invalid_argument );

	// Fewer points than min_pts: everything is unreached (core-distance UNDEFINED).
	std::vector<P2> few = { { 0, 0 }, { 1, 0 }, { 0, 1 } };
	const auto rd_few = optics::compute_reachability_dists( few, 10, 5.0 );
	CHECK( rd_few.size() == 3 );
	bool all_undefined = true;
	for ( const auto& r : rd_few ) { if ( r.reach_dist >= 0.0 ) all_undefined = false; }
	CHECK( all_undefined );

	// All-identical points: auto-epsilon must not collapse to a zero radius. The knee
	// estimator defers to the uniform scale on zero-variance input, so they group into one.
	std::vector<P2> identical( 20, P2{ 3.0, 3.0 } );
	CHECK( optics::epsilon_estimation_knee( identical, 5 ) > 0.0 );
	const auto rd_id = optics::compute_reachability_dists( identical, 5 );
	CHECK( rd_id.size() == 20 );
	const auto cl_id = optics::get_cluster_indices( rd_id, 1.0 );
	CHECK( cl_id.size() == 1 );
	CHECK( cl_id[0].size() == 20 );

	// Collinear (y constant, d_eff == 1): auto-epsilon must stay positive (a degenerate
	// volume would give 0). With a generating distance covering the within-group spacing the
	// three x-axis groups separate.
	std::vector<P2> line;
	for ( int g = 0; g < 3; ++g ) {
		for ( int i = 0; i < 10; ++i ) { line.push_back( { g * 30.0 + i * 0.5, 0.0 } ); }
	}
	CHECK( optics::epsilon_estimation_knee( line, 4 ) > 0.0 );
	const auto rd_line = optics::compute_reachability_dists( line, 4, 3.0 );
	const auto cl_line = optics::get_cluster_indices( rd_line, 5.0 );
	std::size_t big_line = 0;
	for ( const auto& c : cl_line ) { if ( c.size() >= 5 ) ++big_line; }
	CHECK( big_line == 3 );

	// float vs double produce the same cluster structure.
	const std::vector<P2> centers = { { 0, 0 }, { 40, 0 }, { 20, 35 } };
	const auto pd = optics::testdata::gaussian_blobs<double, 2>( centers, 40, 1.5 );
	const auto pf = optics::convert_cloud<float>( pd );
	const auto large = []( const std::vector<std::vector<std::size_t>>& cs ) {
		std::size_t k = 0;
		for ( const auto& c : cs ) { if ( c.size() >= 20 ) ++k; }
		return k;
	};
	const auto big_d = large( optics::cluster_threshold( pd, 5, 12.0 ) );
	const auto big_f = large( optics::cluster_threshold( pf, 5, 12.0 ) );
	CHECK( big_d == 3 );
	CHECK( big_d == big_f );

	// Determinism: identical results across repeated multi-threaded runs.
	const auto r1 = optics::compute_reachability_dists( pd, 5, 12.0, optics::NeighborMode::Precompute, 8 );
	const auto r2 = optics::compute_reachability_dists( pd, 5, 12.0, optics::NeighborMode::Precompute, 8 );
	CHECK( r1 == r2 );
}

TEST_CASE("convenience: convert_cloud + cluster_threshold + extract_xi") {
	// Integer cloud (e.g. color-ish data) -> float, then a one-call flat cut.
	std::vector<std::array<int, 2>> int_pts = {
		{ 100,100 },{ 102,100 },{ 101,101 }, { 0,0 },{ 1,0 },{ 0,1 }, { -100,-100 },{ -102,-100 },{ -101,-101 } };
	auto cloud = optics::convert_cloud<float>( int_pts );
	auto clusters = optics::cluster_threshold( cloud, 2, 10.0, 5.0 );  // explicit eps: 3 tiny far-apart clusters
	CHECK( clusters.size() == 3 );

	// New points-based one-call helpers (min_pts 2nd, threshold/chi 3rd & optional).
	auto xi_pts = optics::extract_xi( cloud, 2, 0.05 );
	CHECK( !xi_pts.empty() );
	// Auto threshold (not supplied) still produces a valid partition of all points.
	auto auto_cut = optics::cluster_threshold( cloud, 2 );
	std::size_t covered = 0;
	for ( const auto& c : auto_cut ) { covered += c.size(); }
	CHECK( covered == cloud.size() );

	// Low-level Xi from a hand-crafted cluster-ordering (what extract_xi runs internally).
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,10.0 },{ 2,9.0 },{ 3,9.0 },{ 4,5.0 },{ 5,5.49 },{ 6,5.0 },
		{ 7,6.5 },{ 8,3.0 },{ 9,2.9 },{ 10,2.8 },{ 11,10.0 },{ 12,12.0 } };
	auto xi = optics::get_cluster_indices( reach_dists, optics::get_chi_clusters_flat( reach_dists, 0.1, 4 ) );
	CHECK( xi.size() == 3 );
}


TEST_CASE("chi_tree_to_points maps ranges to point indices, preserving nesting") {
	// Same hand-crafted ordering as chi_cluster_tree_tests_1: one tree, root {0,11}
	// with children {2,5} and {6,10}. Position i in reach_dists has point_index i+1.
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,10.0 },{ 2,9.0 },{ 3,9.0 },{ 4,5.0 },
		{ 5,5.49 },{ 6,5.0 },
		{ 7,6.5 },
		{ 8,3.0 },
		{ 9,2.9 },{ 10,2.8 },
		{ 11,10.0 },{ 12,12.0 }
	};
	auto trees = optics::get_chi_clusters( reach_dists, 0.1, 4 );
	REQUIRE( trees.size() == 1 );

	auto point_trees = optics::chi_tree_to_points( trees, reach_dists );
	REQUIRE( point_trees.size() == 1 );
	const auto& root = point_trees.front().get_root();

	// Root spans the whole {0,11} range -> point indices 1..12 (parent includes children).
	CHECK( ( root.get_data() == std::vector<std::size_t>( { 1,2,3,4,5,6,7,8,9,10,11,12 } ) ) );
	REQUIRE( root.get_children().size() == 2 );
	CHECK( ( root.get_children()[0].get_data() == std::vector<std::size_t>( { 3,4,5,6 } ) ) );
	CHECK( ( root.get_children()[1].get_data() == std::vector<std::size_t>( { 7,8,9,10,11 } ) ) );

	// Single-tree overload agrees with the forest overload.
	auto one = optics::chi_tree_to_points( trees.front(), reach_dists );
	CHECK( ( one.get_root().get_data() == root.get_data() ) );
}


TEST_CASE("min_cluster_frac filters small clusters to noise (cluster_labels)") {
	using point = std::array<double, 2>;
	// Three well-separated blobs of sizes 3, 9, 3 (15 points total).
	std::vector<point> points = {
		{ 100,100 },{ 102,100 },{ 101,101 },                                   // blob A (3)
		{ 0,0 },{ 1,0 },{ 0,1 },{ 1,1 },{ 2,0 },{ 0,2 },{ 2,1 },{ 1,2 },{ 2,2 },// blob B (9)
		{ -100,-100 },{ -102,-100 },{ -101,-101 }                              // blob C (3)
	};
	const std::size_t n = points.size();
	auto reach = optics::compute_reachability_dists( points, 2, 10.0 );
	auto clusters = optics::get_cluster_indices( reach, 10.0 );
	REQUIRE( clusters.size() == 3 );

	// min_cluster_frac = 1/3 -> min_size = 5: only the 9-point blob survives; the two
	// 3-point blobs are filtered to noise (label -1).
	const std::size_t min_size = static_cast<std::size_t>( ( 1.0 / 3.0 ) * static_cast<double>( n ) );
	CHECK( min_size == 5 );
	auto labels = optics::io::cluster_labels( n, clusters, min_size );
	long long max_label = -1;
	for ( const long long l : labels ) { max_label = std::max( max_label, l ); }
	CHECK( max_label == 0 );  // exactly one surviving cluster
	for ( const auto& c : clusters ) {
		const bool kept = c.size() >= min_size;
		for ( const std::size_t idx : c ) { CHECK( ( labels[idx] >= 0 ) == kept ); }
	}

	// A min size larger than any cluster makes every point noise.
	auto all_noise = optics::io::cluster_labels( n, clusters, n );
	for ( const long long l : all_noise ) { CHECK( l == -1 ); }

	// The same size filter composes with the Xi extractor's flattened output: an
	// impossible min size leaves nothing labeled.
	auto xi = optics::extract_xi( points, 2, 0.05 );
	auto xi_labels = optics::io::cluster_labels( n, xi, n + 1 );
	for ( const long long l : xi_labels ) { CHECK( l == -1 ); }
}


TEST_CASE("Precompute pre-allocation guard: throws over the cap, OnDemand unaffected") {
	const std::vector<std::array<double, 2>> centers = { { 0, 0 }, { 50, 0 } };
	const auto points = optics::testdata::gaussian_blobs<double, 2>( centers, 100, 2.0 );

	// Precompute with a 1-byte cap: the estimate dwarfs it -> throw suggesting OnDemand.
	CHECK_THROWS_AS(
		optics::compute_reachability_dists( points, 5, 5.0, optics::NeighborMode::Precompute, 1,
											 optics::CoreDistMode::Scan, std::size_t{ 1 } ),
		std::runtime_error );

	// A generous cap does not throw and yields the same ordering as the default (no cap).
	const auto capped = optics::compute_reachability_dists( points, 5, 5.0, optics::NeighborMode::Precompute, 1,
														   optics::CoreDistMode::Scan, std::size_t{ 1 } << 40 );
	const auto uncapped = optics::compute_reachability_dists( points, 5, 5.0, optics::NeighborMode::Precompute );
	CHECK( ( capped == uncapped ) );

	// OnDemand has no such buffer, so even a 1-byte cap is ignored.
	const auto on_demand = optics::compute_reachability_dists( points, 5, 5.0, optics::NeighborMode::OnDemand, 0,
															  optics::CoreDistMode::Scan, std::size_t{ 1 } );
	CHECK( on_demand.size() == points.size() );
}


TEST_CASE("epsilon_estimation_knee: opt-in k-distance knee estimator") {
	// Two dense, well-separated blobs (each >> min_pts), so every point's k-distance is
	// the small within-blob scale -- the knee sits there, far below the 100-unit gap.
	const std::vector<std::array<double, 2>> centers = { { 0, 0 }, { 100, 0 } };
	const auto points = optics::testdata::gaussian_blobs<double, 2>( centers, 50, 1.0 );

	const double eps = optics::epsilon_estimation_knee( points, 5 );
	CHECK( eps > 0.0 );
	CHECK( std::isfinite( eps ) );
	CHECK( eps < 50.0 );  // within-cluster scale, well below the inter-blob gap

	// It is a usable generating distance (the ordering is well-formed with it).
	const auto reach = optics::compute_reachability_dists( points, 5, eps );
	CHECK( reach.size() == points.size() );

	// Degenerate: fewer points than min_pts falls back to the uniform estimator.
	const std::vector<std::array<double, 2>> few = { { 0, 0 }, { 1, 0 } };
	CHECK( std::isfinite( optics::epsilon_estimation_knee( few, 5 ) ) );
}


// Generic OPTICS-output invariants over a cluster-ordering. Written against the
// reachability_dist contract (not a specific algorithm) so approximate variants
// (e.g. sOPTICS, #50) can reuse them instead of duplicating.
static void check_ordering_invariants( const std::vector<optics::reachability_dist>& reach, std::size_t n ) {
	CHECK( reach.size() == n );
	if ( n == 0 ) { return; }
	CHECK( reach.front().reach_dist < 0.0 );  // first entry is always UNDEFINED
	std::vector<char> seen( n, 0 );
	for ( const auto& r : reach ) {
		// reachability is either UNDEFINED (-1) or a finite, non-negative distance.
		const bool ok = ( r.reach_dist < 0.0 ) || ( r.reach_dist >= 0.0 && std::isfinite( r.reach_dist ) );
		CHECK( ok );
		REQUIRE( r.point_index < n );
		CHECK( seen[r.point_index] == 0 );  // each point appears exactly once
		seen[r.point_index] = 1;
	}
	std::size_t covered = 0;
	for ( const char c : seen ) { covered += static_cast<std::size_t>( c ); }
	CHECK( covered == n );  // the ordering is a permutation of 0..n-1
}

template <std::size_t Dim>
static void fuzz_invariants_for_dim( unsigned n_seeds ) {
	for ( unsigned s = 0; s < n_seeds; ++s ) {
		const std::size_t n = 25 + s * 13;
		const auto pts = optics::testdata::uniform_noise<double, Dim>( n, 0.0, 100.0, 100u + s );
		for ( const std::size_t min_pts : { std::size_t{ 2 }, std::size_t{ 6 } } ) {
			const auto reach = optics::compute_reachability_dists( pts, min_pts );
			check_ordering_invariants( reach, n );

			// A flat threshold cut is a full partition: every point exactly once.
			const auto cut = optics::get_cluster_indices( reach, optics::detail::default_threshold( reach ) );
			std::vector<int> count( n, 0 );
			for ( const auto& c : cut ) {
				for ( const std::size_t idx : c ) { REQUIRE( idx < n ); count[idx]++; }
			}
			for ( const int k : count ) { CHECK( k == 1 ); }

			// Xi flat clusters: valid (begin<=end<n) ranges; extracted index lists in-range.
			const auto flat = optics::get_chi_clusters_flat( reach, 0.05, min_pts );
			for ( const auto& rng : flat ) {
				CHECK( rng.first <= rng.second );
				REQUIRE( rng.second < reach.size() );
			}
			const auto xi = optics::get_cluster_indices( reach, flat );
			for ( const auto& c : xi ) {
				CHECK( !c.empty() );
				for ( const std::size_t idx : c ) { CHECK( idx < n ); }
			}
		}
	}
}

TEST_CASE("property/fuzz: OPTICS-output invariants hold over random clouds") {
	fuzz_invariants_for_dim<2>( 8 );
	fuzz_invariants_for_dim<3>( 6 );
	fuzz_invariants_for_dim<16>( 2 );  // a couple of high-D cases for breadth

	// Also exercise structured (blobby) clouds, not just uniform noise.
	const std::vector<std::array<double, 2>> centers = { { 0, 0 }, { 40, 0 }, { 20, 35 } };
	const auto blobs = optics::testdata::gaussian_blobs<double, 2>( centers, 80, 1.5 );
	const auto reach = optics::compute_reachability_dists( blobs, 5 );
	check_ordering_invariants( reach, blobs.size() );

	// Empty cloud: a well-formed empty ordering.
	const std::vector<std::array<double, 2>> empty;
	check_ordering_invariants( optics::compute_reachability_dists( empty, 3 ), 0 );
}


// Sum of all eps-neighborhood sizes (what Precompute caches) and the largest single
// neighborhood (what OnDemand holds at a time), for a cloud read straight from the backend.
template <std::size_t Dim>
static std::pair<std::size_t, std::size_t> neighbor_cache_sizes( std::size_t n, double box, double eps, unsigned seed ) {
	const auto pts = optics::testdata::uniform_noise<double, Dim>( n, 0.0, box, seed );
	const optics::NanoflannBackend<double, Dim> backend( pts );
	std::size_t total = 0, max_one = 0;
	std::vector<std::size_t> buf;
	for ( const auto& p : pts ) {
		buf.clear();
		backend.radius_search( p, eps, buf );
		total += buf.size();
		max_one = std::max( max_one, buf.size() );
	}
	return { total, max_one };
}

TEST_CASE("memory invariant: Precompute cache grows with n; OnDemand holds one neighborhood") {
	// Two clouds at the SAME point density (box area scaled with n), so the average
	// neighborhood is constant. The Precompute cache (sum of all neighborhoods) then grows
	// ~linearly with n, while the OnDemand footprint (one neighborhood) stays bounded. This
	// is the data-structure invariant that motivated the OnDemand default; asserting the
	// sizes directly is deterministic and platform-independent (no peak-RSS sampling).
	const double eps = 5.0;
	const std::size_t n1 = 2000, n2 = 8000;       // 4x the points
	const double box1 = 50.0, box2 = 100.0;       // 4x the area (2-D) -> same density

	const auto [pre1, od1] = neighbor_cache_sizes<2>( n1, box1, eps, 11u );
	const auto [pre2, od2] = neighbor_cache_sizes<2>( n2, box2, eps, 12u );

	// OnDemand holds a single neighborhood: tiny next to the full Precompute cache.
	CHECK( od1 * 40 < pre1 );
	CHECK( od2 * 40 < pre2 );

	// Precompute cache scales ~linearly with n (4x points -> ~4x cached entries).
	const double pre_ratio = static_cast<double>( pre2 ) / static_cast<double>( pre1 );
	CHECK( pre_ratio > 2.5 );
	CHECK( pre_ratio < 6.0 );

	// OnDemand's largest neighborhood stays bounded at fixed density (no ~4x growth).
	const double od_ratio = static_cast<double>( od2 ) / static_cast<double>( od1 );
	CHECK( od_ratio < 2.5 );
}


TEST_CASE("ceos_neighbors out_sq: squared distances exact + parallel, neighbor lists unchanged (#55 sOPTICS reuse)") {
	auto pts = optics::testdata::make_blobs<double, 3>( 4, 100, 30.0, 1.0, 17u );
	for ( auto& p : pts ) {
		const double s = std::sqrt( p[0] * p[0] + p[1] * p[1] + p[2] * p[2] );
		if ( s > 0.0 ) { for ( auto& c : p ) { c /= s; } }
	}
	const std::size_t n = pts.size();
	optics::detail::CeosParams params;
	params.n_projections = 256;
	params.k = 16;
	params.seed = 3;

	std::vector<std::vector<double>> nsq;
	const auto nbrs = optics::detail::ceos_neighbors( pts, 0.3, 5, params, &nsq );
	const auto nbrs_plain = optics::detail::ceos_neighbors( pts, 0.3, 5, params );  // no out_sq

	REQUIRE( nbrs.size() == n );
	REQUIRE( nsq.size() == n );
	for ( std::size_t q = 0; q < n; ++q ) {
		CHECK( ( nbrs[q] == nbrs_plain[q] ) );        // out_sq must not change the neighbor list
		REQUIRE( nsq[q].size() == nbrs[q].size() );   // distances run parallel to indices
		for ( std::size_t t = 0; t < nbrs[q].size(); ++t ) {
			// Each reused squared distance is BIT-IDENTICAL to detail::square_dist -- the
			// basis for sOPTICS reusing them in core-dist + relax without changing the ordering.
			CHECK( nsq[q][t] == optics::detail::square_dist( pts[q], pts[nbrs[q][t]] ) );
		}
	}
}


TEST_CASE("ceos_neighbors: exact precision, good recall, symmetric, self-matching") {
	// Four well-separated blobs, L2-normalized onto the unit sphere (cosine metric).
	auto pts = optics::testdata::make_blobs<double, 3>( 4, 120, 30.0, 1.0, 123u );
	for ( auto& p : pts ) {
		const double nrm = std::sqrt( p[0] * p[0] + p[1] * p[1] + p[2] * p[2] );
		if ( nrm > 0.0 ) { for ( auto& c : p ) { c /= nrm; } }
	}
	const std::size_t n = pts.size();
	const double eps = 0.25;  // Euclidean radius on the unit sphere (monotone in cosine distance)
	const std::size_t min_pts = 5;

	optics::detail::CeosParams params;
	params.n_projections = 512;
	params.k = 20;
	params.m = 40;
	params.seed = 7;
	const auto approx = optics::detail::ceos_neighbors( pts, eps, min_pts, params );
	REQUIRE( approx.size() == n );

	// Brute-force true eps-neighborhoods (incl. self) for precision/recall.
	const double eps_sq = eps * eps;
	std::vector<std::set<std::size_t>> truth( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		for ( std::size_t j = 0; j < n; ++j ) {
			if ( optics::detail::square_dist( pts[i], pts[j] ) <= eps_sq ) { truth[i].insert( j ); }
		}
	}

	std::size_t total_true = 0, found_true = 0, returned = 0, returned_in_truth = 0;
	for ( std::size_t i = 0; i < n; ++i ) {
		CHECK( std::find( approx[i].begin(), approx[i].end(), i ) != approx[i].end() );  // self-match
		const std::set<std::size_t> a( approx[i].begin(), approx[i].end() );
		CHECK( a.size() == approx[i].size() );  // no duplicates
		for ( const std::size_t x : a ) { returned++; if ( truth[i].count( x ) ) { returned_in_truth++; } }
		for ( const std::size_t t : truth[i] ) { total_true++; if ( a.count( t ) ) { found_true++; } }
	}

	// Precision is exact by construction: every returned candidate is within eps.
	CHECK( returned == returned_in_truth );
	// Recall: CEOs recovers the large majority of true neighbors on well-separated data.
	const double recall = static_cast<double>( found_true ) / static_cast<double>( total_true );
	CHECK( recall > 0.7 );

	// Symmetry: x in N(q)  <=>  q in N(x).
	for ( std::size_t q = 0; q < n; ++q ) {
		for ( const std::size_t x : approx[q] ) {
			CHECK( std::find( approx[x].begin(), approx[x].end(), q ) != approx[x].end() );
		}
	}
}


// Rand index between two flat labelings: the fraction of point-pairs that agree on
// same/different cluster. 1.0 == identical partition.
static double rand_index( const std::vector<long long>& a, const std::vector<long long>& b ) {
	const std::size_t n = a.size();
	std::size_t agree = 0, total = 0;
	for ( std::size_t i = 0; i < n; ++i ) {
		for ( std::size_t j = i + 1; j < n; ++j ) {
			if ( ( a[i] == a[j] ) == ( b[i] == b[j] ) ) { ++agree; }
			++total;
		}
	}
	return total ? static_cast<double>( agree ) / static_cast<double>( total ) : 1.0;
}

// Per-point labels from a cluster list (each cluster -- including singletons -- a distinct id).
static std::vector<long long> labels_from_clusters( std::size_t n, const std::vector<std::vector<std::size_t>>& clusters ) {
	std::vector<long long> labels( n, -1 );
	long long id = 0;
	for ( const auto& c : clusters ) {
		for ( const std::size_t idx : c ) { labels[idx] = id; }
		++id;
	}
	return labels;
}

TEST_CASE("sOPTICS: high Rand-index agreement with exact OPTICS; seed-deterministic") {
	// Normalize so both algorithms share the SAME metric: exact OPTICS on the unit
	// sphere is monotone-equivalent to cosine OPTICS, which sOPTICS approximates.
	auto pts = optics::testdata::make_blobs<double, 3>( 5, 120, 30.0, 1.0, 321u );
	for ( auto& p : pts ) {
		const double nrm = std::sqrt( p[0] * p[0] + p[1] * p[1] + p[2] * p[2] );
		if ( nrm > 0.0 ) { for ( auto& c : p ) { c /= nrm; } }
	}
	const std::size_t n = pts.size();
	const std::size_t min_pts = 6;
	const double eps = 0.3;

	const auto exact = optics::compute_reachability_dists( pts, min_pts, eps );
	const auto approx = optics::compute_soptics_reachability_dists( pts, min_pts, eps, 512u, 20u, std::size_t{ 40 }, 7u );

	// The approximate ordering still satisfies the generic OPTICS-output invariants (B4).
	check_ordering_invariants( exact, n );
	check_ordering_invariants( approx, n );

	// Flat-cut both plots at the same threshold; the partitions should largely agree.
	const double thr = 0.5 * eps;
	const auto la = labels_from_clusters( n, optics::get_cluster_indices( exact, thr ) );
	const auto lb = labels_from_clusters( n, optics::get_cluster_indices( approx, thr ) );
	CHECK( rand_index( la, lb ) > 0.9 );  // typically ~0.99 on well-separated data

	// Determinism: same seed => byte-identical ordering.
	const auto approx_same = optics::compute_soptics_reachability_dists( pts, min_pts, eps, 512u, 20u, std::size_t{ 40 }, 7u );
	CHECK( ( approx == approx_same ) );

	// A different seed still agrees closely with exact OPTICS (stability).
	const auto approx_other = optics::compute_soptics_reachability_dists( pts, min_pts, eps, 512u, 20u, std::size_t{ 40 }, 99u );
	const auto ld = labels_from_clusters( n, optics::get_cluster_indices( approx_other, thr ) );
	CHECK( rand_index( la, ld ) > 0.9 );
}

TEST_CASE("sOPTICS: edge cases (empty, < min_pts, identical points, high-D)") {
	// Empty cloud -> well-formed empty ordering.
	const std::vector<std::array<double, 3>> empty;
	CHECK( optics::compute_soptics_reachability_dists( empty, 4 ).empty() );

	// Fewer points than min_pts: a valid permutation, every reach UNDEFINED.
	std::vector<std::array<double, 3>> few = { { 1, 0, 0 }, { 0, 1, 0 } };
	const auto r_few = optics::compute_soptics_reachability_dists( few, 5, -1.0, 64u );
	CHECK( r_few.size() == few.size() );
	for ( const auto& rd : r_few ) { CHECK( rd.reach_dist < 0.0 ); }

	// Identical points (degenerate projections) must not crash and stay a permutation.
	const std::vector<std::array<double, 3>> same( 30, { 1.0, 2.0, 3.0 } );
	const auto r_same = optics::compute_soptics_reachability_dists( same, 4, -1.0, 64u );
	CHECK( r_same.size() == same.size() );

	// High-D smoke test.
	const auto hd = optics::testdata::make_blobs<double, 16>( 3, 40, 20.0, 1.0, 5u );
	const auto r_hd = optics::compute_soptics_reachability_dists( hd, 5, -1.0, 256u );
	CHECK( r_hd.size() == hd.size() );
}


TEST_CASE("sOPTICS metrics (#51): L2/L1 random-feature embeddings recover Euclidean clustering") {
	// Raw (un-normalized) Euclidean blobs: the metric path must recover the L2/L1 structure
	// of the ORIGINAL data, not the cosine geometry of the cosine default.
	const auto pts = optics::testdata::make_blobs<double, 4>( 5, 120, 30.0, 1.0, 321u );
	const std::size_t n = pts.size();
	const std::size_t min_pts = 6;

	// Reference: exact Euclidean OPTICS. Extract via each ordering's OWN high-percentile
	// threshold -- sOPTICS reachabilities live in feature-cosine units, exact OPTICS' in
	// Euclidean units, so a per-ordering threshold makes the partitions comparable.
	const auto exact = optics::compute_reachability_dists( pts, min_pts );
	const auto exact_lbl = labels_from_clusters( n, optics::get_cluster_indices( exact, optics::detail::default_threshold( exact ) ) );

	for ( const optics::Metric metric : { optics::Metric::L2, optics::Metric::L1 } ) {
		const auto approx = optics::compute_soptics_reachability_dists(
			pts, min_pts, -1.0, 1024u, 0u, std::size_t{ 0 }, 7u, 0u, metric );
		check_ordering_invariants( approx, n );
		const auto approx_lbl = labels_from_clusters( n, optics::get_cluster_indices( approx, optics::detail::default_threshold( approx ) ) );
		CHECK( rand_index( exact_lbl, approx_lbl ) > 0.85 );  // both metrics separate these blobs

		// Deterministic in seed.
		const auto approx_same = optics::compute_soptics_reachability_dists(
			pts, min_pts, -1.0, 1024u, 0u, std::size_t{ 0 }, 7u, 0u, metric );
		CHECK( ( approx == approx_same ) );
	}

	// An explicit kernel bandwidth is accepted and yields a well-formed ordering.
	const auto fixed = optics::compute_soptics_reachability_dists(
		pts, min_pts, -1.0, 512u, 0u, std::size_t{ 0 }, 1u, 0u, optics::Metric::L2, 25.0 );
	check_ordering_invariants( fixed, n );
}


TEST_CASE("Xi min_cluster_size: decoupled from min_pts, default-preserving (#57)") {
	const auto pts = optics::testdata::make_blobs<double, 2>( 5, 60, 30.0, 1.0, 3u );
	const std::size_t min_pts = 10;
	const auto reach = optics::compute_reachability_dists( pts, min_pts );

	// min_cluster_size 0 (default) resolves to min_pts -> identical to passing min_pts
	// explicitly, so existing behavior (and the chi_test_* cases) is unchanged.
	const auto base = optics::get_chi_clusters_flat( reach, 0.05, min_pts );
	const auto same = optics::get_chi_clusters_flat( reach, 0.05, min_pts, 0.0, min_pts );
	CHECK( ( base == same ) );

	// A different (smaller) min_cluster_size is accepted and yields valid ranges.
	const auto smaller = optics::get_chi_clusters_flat( reach, 0.05, min_pts, 0.0, 3 );
	for ( const auto& c : smaller ) { CHECK( c.first <= c.second ); REQUIRE( c.second < reach.size() ); }
	// get_chi_clusters (tree) plumbs the same parameter.
	CHECK( !optics::get_chi_clusters( reach, 0.05, min_pts, 0.0, 3 ).empty() );
}


TEST_CASE("knee epsilon: smaller than uniform on clustered data, yields good Xi clusters (#57)") {
	// 15 tight blobs. make_blobs emits blob b's points contiguously, so point i's ground-
	// truth label is i / points_per_blob. The under-segmentation that motivated #57 is the
	// uniform-density epsilon_estimation over-shooting on clustered data and over-smoothing
	// the reachability; the k-distance-knee estimator lands near the within-cluster scale.
	// (The dramatic R15 recovery is shown in docs/benchmarking.md; that data is not bundled.)
	const std::size_t n_blobs = 15, per = 40, min_pts = 10;
	const auto pts = optics::testdata::make_blobs<double, 2>( n_blobs, per, 60.0, 0.4, 11u );
	const std::size_t n = pts.size();
	std::vector<long long> truth( n );
	for ( std::size_t i = 0; i < n; ++i ) { truth[i] = static_cast<long long>( i / per ); }

	const double eps_uniform = optics::epsilon_estimation( pts, min_pts );
	const double eps_knee = optics::epsilon_estimation_knee( pts, min_pts );
	CHECK( eps_knee < eps_uniform );  // the mechanism behind #57
	CHECK( eps_knee > 0.0 );

	// The knee eps produces a valid, high-quality Xi clustering of the tight blobs.
	const auto reach = optics::compute_reachability_dists( pts, min_pts, eps_knee );
	const auto labels = labels_from_clusters(
		n, optics::get_cluster_indices( reach, optics::get_chi_clusters_flat( reach, 0.05, min_pts ) ) );
	CHECK( rand_index( truth, labels ) > 0.9 );
}


TEST_CASE("backend matrix: clustering is consistent across neighbor-search backends") {
	static const std::size_t N = 2;
	const std::vector<std::array<double, N>> centers = { { 0, 0 }, { 60, 0 }, { 30, 50 } };
	const auto points = optics::testdata::gaussian_blobs<double, N>( centers, 150, 1.5 );

	// Baseline: exact nanoflann -> a valid ordering with the three dense clusters.
	const auto reach_nano = optics::compute_reachability_dists<double, N>( points, 10, 10.0 );
	check_ordering_invariants( reach_nano, points.size() );
	const auto clusters_nano = optics::get_cluster_indices( reach_nano, 10.0 );
	std::size_t big_nano = 0;
	for ( const auto& c : clusters_nano ) { if ( c.size() >= 50 ) { ++big_nano; } }
	CHECK( big_nano == 3 );

	// Approximate nanoflann: still a valid ordering recovering the same three clusters
	// (low-D recall ~1.0). Not asserted bit-equal -- it is, by design, an approximation.
	const auto reach_approx =
		optics::compute_reachability_dists<double, N, optics::ApproxNanoflannBackend<double, N>>( points, 10, 10.0 );
	check_ordering_invariants( reach_approx, points.size() );
	const auto clusters_approx = optics::get_cluster_indices( reach_approx, 10.0 );
	std::size_t big_approx = 0;
	for ( const auto& c : clusters_approx ) { if ( c.size() >= 50 ) { ++big_approx; } }
	CHECK( big_approx == 3 );

#ifdef OPTICS_ENABLE_BOOST_RTREE
	// The exact Boost backend must match exact nanoflann bit-for-bit.
	const auto reach_boost =
		optics::compute_reachability_dists<double, N, optics::BoostRTreeBackend<double, N>>( points, 10, 10.0 );
	CHECK( ( reach_nano == reach_boost ) );
#endif
}


TEST_CASE("RadiusSearchWithDists: same neighbors as radius_search, exact squared distances (#55)") {
	const auto pts = optics::testdata::make_blobs<double, 3>( 3, 80, 20.0, 1.0, 9u );
	const optics::NanoflannBackend<double, 3> backend( pts );
	const double r = 6.0;
	for ( std::size_t i = 0; i < pts.size(); i += 7 ) {
		std::vector<std::size_t> a, b;
		std::vector<double> b_sq;
		backend.radius_search( pts[i], r, a );
		backend.radius_search_with_dists( pts[i], r, b, b_sq );
		REQUIRE( b.size() == b_sq.size() );
		CHECK( ( sorted( a ) == sorted( b ) ) );  // same neighbor set
		// Each reused squared distance is BIT-IDENTICAL to detail::square_dist -- the basis
		// for the #55 fast path leaving the OPTICS ordering byte-for-byte unchanged.
		for ( std::size_t t = 0; t < b.size(); ++t ) {
			CHECK( b_sq[t] == optics::detail::square_dist( pts[i], pts[b[t]] ) );
		}
	}
	// The capability is modeled for double (where the reuse is exact) but not for float.
	CHECK( ( optics::RadiusSearchWithDists<optics::NanoflannBackend<double, 3>, double, 3> ) );
	CHECK( ( !optics::RadiusSearchWithDists<optics::NanoflannBackend<float, 3>, float, 3> ) );
}


#ifdef OPTICS_ENABLE_HNSW
// Only built when the optional HNSW backend is enabled (-DOPTICS_ENABLE_HNSW=ON). HNSW is
// APPROXIMATE, so this asserts high (not exact) recall vs nanoflann in high-D and that an
// end-to-end OPTICS run over it recovers the same well-separated clusters (#47).
TEST_CASE("HnswBackend: high-D approximate recall + usable OPTICS clustering (#47)") {
	constexpr std::size_t Dim = 16;
	const auto pts = optics::testdata::make_blobs<double, Dim>( 5, 200, 30.0, 1.0, 71u );
	const std::size_t n = pts.size();

	// Satisfies the NeighborSearch concept (+ the optional KnnCoreDist capability).
	CHECK( ( optics::NeighborSearch<optics::HnswBackend<double, Dim>, double, Dim> ) );
	CHECK( ( optics::KnnCoreDist<optics::HnswBackend<double, Dim>, double, Dim> ) );

	const optics::NanoflannBackend<double, Dim> exact( pts );
	const optics::HnswBackend<double, Dim> hnsw( pts, 32, 400 );  // higher M / ef => better recall

	// Neighbor-set recall vs exact at a radius covering within-blob neighborhoods. Returned
	// neighbors are within r (up to float rounding) -- the radius filter is exact-ish.
	const double r = 6.0;
	const double r_sq = r * r;
	std::size_t total_true = 0, found = 0;
	for ( std::size_t i = 0; i < n; i += 5 ) {
		std::vector<std::size_t> ex, ap;
		exact.radius_search( pts[i], r, ex );
		hnsw.radius_search( pts[i], r, ap );
		const std::set<std::size_t> aps( ap.begin(), ap.end() );
		for ( const std::size_t t : ex ) { total_true++; if ( aps.count( t ) ) { found++; } }
		for ( const std::size_t a : ap ) { CHECK( optics::detail::square_dist( pts[i], pts[a] ) <= r_sq * 1.02 + 1e-3 ); }
	}
	const double recall = total_true ? static_cast<double>( found ) / static_cast<double>( total_true ) : 1.0;
	CHECK( recall > 0.85 );

	// knn_core_dist agrees closely with the exact backend's (HNSW's native operation).
	const auto cd_exact = exact.knn_core_dist( pts[0], 8, r );
	const auto cd_hnsw = hnsw.knn_core_dist( pts[0], 8, r );
	REQUIRE( cd_exact.has_value() );
	REQUIRE( cd_hnsw.has_value() );
	CHECK( *cd_hnsw == doctest::Approx( *cd_exact ).epsilon( 0.05 ) );

	// End-to-end: OPTICS over the (default-parameter) HNSW backend recovers the same blobs.
	const auto reach_exact = optics::compute_reachability_dists<double, Dim>( pts, 8, r );
	const auto reach_hnsw =
		optics::compute_reachability_dists<double, Dim, optics::HnswBackend<double, Dim>>( pts, 8, r );
	check_ordering_invariants( reach_hnsw, n );
	const double thr = 0.5 * r;
	const auto le = labels_from_clusters( n, optics::get_cluster_indices( reach_exact, thr ) );
	const auto lh = labels_from_clusters( n, optics::get_cluster_indices( reach_hnsw, thr ) );
	CHECK( rand_index( le, lh ) > 0.9 );

	std::cout << "HnswBackend tests successful! (recall=" << recall << ")" << std::endl;
}
#endif


//=== Weighted / unique-point OPTICS (issue #46) =============================

TEST_CASE("weighted core-distance: matches unweighted at weight 1; weighted selection (#46)") {
	// Distances 0,1,2,3 (squared 0,1,4,9); neighbor indices 0..3.
	const std::vector<double> sq = { 0.0, 1.0, 4.0, 9.0 };
	const std::vector<std::size_t> nbr = { 0, 1, 2, 3 };

	// All weight 1 == the unweighted min_pts-th-neighbor distance, for every min_pts.
	const std::vector<std::size_t> w1 = { 1, 1, 1, 1 };
	for ( std::size_t mp = 1; mp <= 4; ++mp ) {
		const auto u = optics::detail::compute_core_dist_from_sq( sq, mp );
		const auto w = optics::detail::compute_core_dist_weighted_from_sq( sq, nbr, w1, mp );
		REQUIRE( u.has_value() );
		REQUIRE( w.has_value() );
		CHECK( *w == doctest::Approx( *u ) );
	}

	// Heavy first neighbor (weight 2 at distance 0): min_pts 3 is reached at distance 1.
	const std::vector<std::size_t> w2 = { 2, 1, 1, 1 };
	CHECK( *optics::detail::compute_core_dist_weighted_from_sq( sq, nbr, w2, 3 ) == doctest::Approx( 1.0 ) );

	// Self weight alone satisfies min_pts => core-distance 0.
	const std::vector<std::size_t> w5 = { 5, 1, 1, 1 };
	CHECK( *optics::detail::compute_core_dist_weighted_from_sq( sq, nbr, w5, 3 ) == doctest::Approx( 0.0 ) );

	// Total weight below min_pts => undefined.
	CHECK_FALSE( optics::detail::compute_core_dist_weighted_from_sq( sq, nbr, w1, 5 ).has_value() );

	// Point-based variant agrees with the from-squared variant.
	const std::vector<std::array<double, 1>> pts = { { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 } };
	CHECK( *optics::detail::compute_core_dist_weighted( pts[0], pts, nbr, w2, 3 ) == doctest::Approx( 1.0 ) );
}


TEST_CASE("deduplicate / expand / quantize preprocessing (#46)") {
	std::vector<std::array<int, 3>> int_pts = {
		{ 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 }, { 20, 20, 20 }, { 30, 30, 30 }, { 30, 30, 30 } };
	const auto cloud = optics::convert_cloud<float>( int_pts );
	const auto d = optics::deduplicate( cloud );

	CHECK( d.unique_points.size() == 3 );
	std::size_t sumw = 0;
	for ( const auto w : d.weights ) { sumw += w; }
	CHECK( sumw == cloud.size() );
	// First-seen order: {10},{20},{30} with weights 3,1,2.
	CHECK( d.weights[0] == 3 );
	CHECK( d.weights[1] == 1 );
	CHECK( d.weights[2] == 2 );
	CHECK( d.unique_of_original[0] == 0 );
	CHECK( d.unique_of_original[3] == 1 );
	CHECK( d.unique_of_original[5] == 2 );

	// Expanding unique clusters reconstructs the full original index set, no gaps/dups.
	const std::vector<std::vector<std::size_t>> uniq_clusters = { { 0, 2 }, { 1 } };  // {10,30} and {20}
	const auto expanded = optics::expand_clusters_to_original( uniq_clusters, d.unique_of_original );
	std::set<std::size_t> all;
	for ( const auto& c : expanded ) { for ( const auto i : c ) { all.insert( i ); } }
	CHECK( all.size() == cloud.size() );

	// quantize snaps near-identical values onto a grid so they then deduplicate.
	const std::vector<std::array<float, 3>> ramp = {
		{ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f }, { 3.0f, 3.0f, 3.0f } };
	const auto q = optics::quantize( ramp, 8.0 );  // 0..3 all map to cell centre 4 => identical
	CHECK( optics::deduplicate( q ).unique_points.size() == 1 );
	CHECK( optics::quantize( ramp, 0.0 ).size() == ramp.size() );  // non-positive bin is a no-op
}


TEST_CASE("weighted dedup OPTICS == full OPTICS partition (#46)") {
	// Well-separated blobs, each point replicated 1..4 times to create exact duplicates.
	const auto base = optics::testdata::make_blobs<double, 2>( 4, 60, 50.0, 1.0, 4242u );
	std::vector<std::array<double, 2>> full;
	std::mt19937 rng( 7 );
	for ( const auto& p : base ) {
		const int reps = 1 + static_cast<int>( rng() % 4 );
		for ( int r = 0; r < reps; ++r ) { full.push_back( p ); }
	}
	const std::size_t n = full.size();
	const std::size_t min_pts = 8;
	const double eps = 12.0, thr = 6.0;

	// Explicit eps + threshold so the dedup and full paths share the SAME parameters.
	const auto full_cl = optics::cluster_threshold( full, min_pts, thr, eps, optics::NeighborMode::OnDemand, 0, false );
	const auto dedup_cl = optics::cluster_threshold( full, min_pts, thr, eps, optics::NeighborMode::OnDemand, 0, true );
	CHECK( rand_index( labels_from_clusters( n, full_cl ), labels_from_clusters( n, dedup_cl ) ) > 0.99 );

	// Every original point is labeled exactly once by the dedup path.
	std::size_t covered = 0;
	for ( const auto& c : dedup_cl ) { covered += c.size(); }
	CHECK( covered == n );

	// Xi path: same partition equivalence (exercises the weighted steep-area spans).
	const auto full_xi = optics::extract_xi( full, min_pts, 0.05, eps, optics::NeighborMode::OnDemand, 0, 0.0, 0, false );
	const auto dedup_xi = optics::extract_xi( full, min_pts, 0.05, eps, optics::NeighborMode::OnDemand, 0, 0.0, 0, true );
	CHECK( rand_index( labels_from_clusters( n, full_xi ), labels_from_clusters( n, dedup_xi ) ) > 0.99 );
}


TEST_CASE("weighted OPTICS: validation + empty/unit weights leave results unchanged (#46)") {
	const auto pts = optics::testdata::make_blobs<double, 2>( 3, 40, 40.0, 1.0, 9u );
	const std::size_t n = pts.size();

	// Size mismatch throws.
	const std::vector<std::size_t> bad( n - 1, 1 );
	CHECK_THROWS_AS(
		optics::compute_reachability_dists( pts, 5, 8.0, optics::NeighborMode::OnDemand, 0, optics::CoreDistMode::Scan, 0, bad ),
		std::invalid_argument );

	// Empty weights => byte-identical ordering to the default call.
	const auto base = optics::compute_reachability_dists( pts, 5, 8.0, optics::NeighborMode::OnDemand, 0, optics::CoreDistMode::Scan );
	const auto same = optics::compute_reachability_dists( pts, 5, 8.0, optics::NeighborMode::OnDemand, 0, optics::CoreDistMode::Scan, 0, std::vector<std::size_t>{} );
	CHECK( ( base == same ) );

	// All-ones weights => same partition as unweighted.
	const std::vector<std::size_t> ones( n, 1 );
	const auto w1 = optics::compute_reachability_dists( pts, 5, 8.0, optics::NeighborMode::OnDemand, 0, optics::CoreDistMode::Scan, 0, ones );
	const auto lb = labels_from_clusters( n, optics::get_cluster_indices( base, 4.0 ) );
	const auto lw = labels_from_clusters( n, optics::get_cluster_indices( w1, 4.0 ) );
	CHECK( rand_index( lb, lw ) > 0.999 );
}


TEST_CASE("weighted eps: total-weight estimate matches the expanded full cloud (#46)") {
	const auto unique = optics::testdata::make_blobs<double, 2>( 4, 30, 40.0, 1.0, 13u );
	// Give each unique point a weight and build the explicitly-expanded full cloud.
	std::vector<std::size_t> weights( unique.size() );
	std::vector<std::array<double, 2>> expanded;
	for ( std::size_t i = 0; i < unique.size(); ++i ) {
		weights[i] = 1 + ( i % 3 );
		for ( std::size_t r = 0; r < weights[i]; ++r ) { expanded.push_back( unique[i] ); }
	}
	const double e_weighted = optics::epsilon_estimation( unique, 5, weights );
	const double e_full = optics::epsilon_estimation( expanded, 5 );
	CHECK( e_weighted == doctest::Approx( e_full ) );  // identical box geometry, identical total count
}


TEST_CASE("weighted knee epsilon: within-cluster scale; weighted k-distance (#46)") {
	// Tight blobs, replicated into exact duplicates carrying weights.
	const auto base = optics::testdata::make_blobs<double, 2>( 6, 40, 60.0, 0.5, 31u );
	std::vector<std::array<double, 2>> full;
	std::mt19937 rng( 2 );
	for ( const auto& p : base ) {
		const int reps = 1 + static_cast<int>( rng() % 3 );
		for ( int r = 0; r < reps; ++r ) { full.push_back( p ); }
	}
	const auto d = optics::deduplicate( full );
	const std::size_t mp = 6;

	const double knee = optics::epsilon_estimation_knee<double, 2>( d.unique_points, mp, d.weights );
	const double unif = optics::epsilon_estimation( d.unique_points, mp, d.weights );
	CHECK( knee > 0.0 );
	CHECK( knee < unif );  // knee tracks the within-cluster scale; uniform over-shoots on clustered data

	// Backend: weighted k-distance with all-ones weights == the unweighted k-distance.
	const optics::NanoflannBackend<double, 2> be( d.unique_points );
	const std::vector<std::size_t> ones( d.unique_points.size(), 1 );
	for ( std::size_t i = 0; i < std::min<std::size_t>( 5, d.unique_points.size() ); ++i ) {
		const auto w = be.knn_core_dist_weighted( d.unique_points[i], ones, mp );
		const auto u = be.knn_core_dist( d.unique_points[i], mp, 1e18 );
		REQUIRE( w.has_value() );
		REQUIRE( u.has_value() );
		CHECK( *w == doctest::Approx( *u ) );
	}
}


TEST_CASE("Xi weighted spans: all-ones position weights are identical to unweighted (#46)") {
	// The prefix-sum threading must not perturb the extractor when every weight is 1
	// (this is what keeps the pinned chi_test_* green). Same hand-crafted ordering as the
	// convenience test, plus a second valley shape.
	const std::vector<optics::reachability_dist> reach_a = {
		{ 1, 10.0 }, { 2, 9.0 }, { 3, 9.0 }, { 4, 5.0 }, { 5, 5.49 }, { 6, 5.0 },
		{ 7, 6.5 }, { 8, 3.0 }, { 9, 2.9 }, { 10, 2.8 }, { 11, 10.0 }, { 12, 12.0 } };
	const std::vector<optics::reachability_dist> reach_b = {
		{ 1, 10.0 }, { 2, 10.0 }, { 3, 2.0 }, { 4, 2.0 }, { 5, 2.0 }, { 6, 10.0 }, { 7, 10.0 } };

	for ( const auto* reach : { &reach_a, &reach_b } ) {
		for ( const std::size_t mp : { 2u, 4u } ) {
			const std::vector<std::size_t> ones( reach->size(), 1 );
			const auto unweighted = optics::get_chi_clusters_flat( *reach, 0.1, mp );
			const auto weighted_ones = optics::get_chi_clusters_flat( *reach, 0.1, mp, 0.0, 0, ones );
			CHECK( ( unweighted == weighted_ones ) );
		}
	}
}


TEST_CASE("weighted sOPTICS: empty weights unchanged; dedup agrees with full (#46)") {
	auto pts = optics::testdata::make_blobs<double, 3>( 4, 80, 30.0, 1.0, 555u );
	for ( auto& p : pts ) {  // cosine metric: normalize onto the unit sphere
		const double nrm = std::sqrt( p[0] * p[0] + p[1] * p[1] + p[2] * p[2] );
		if ( nrm > 0.0 ) { for ( auto& c : p ) { c /= nrm; } }
	}
	std::vector<std::array<double, 3>> full;
	std::mt19937 rng( 3 );
	for ( const auto& p : pts ) {
		const int reps = 1 + static_cast<int>( rng() % 3 );
		for ( int r = 0; r < reps; ++r ) { full.push_back( p ); }
	}
	const std::size_t n = full.size();
	const std::size_t min_pts = 6;
	const double eps = 0.3;

	// Empty weights => byte-identical to the unweighted sOPTICS call.
	const auto a = optics::compute_soptics_reachability_dists( full, min_pts, eps, 256u, 16u, std::size_t{ 32 }, 7u );
	const auto a2 = optics::compute_soptics_reachability_dists( full, min_pts, eps, 256u, 16u, std::size_t{ 32 }, 7u, 0u, optics::Metric::Cosine, 0.0, std::vector<std::size_t>{} );
	CHECK( ( a == a2 ) );

	// Weighted dedup vs full: the flat-cut partitions agree (approximate => statistical).
	const auto d = optics::deduplicate( full );
	const auto wreach = optics::compute_soptics_reachability_dists(
		d.unique_points, min_pts, eps, 256u, 16u, std::size_t{ 32 }, 7u, 0u, optics::Metric::Cosine, 0.0, d.weights );
	const double thr = 0.5 * eps;
	const auto full_lbl = labels_from_clusters( n, optics::get_cluster_indices( a, thr ) );
	const auto exp = optics::expand_clusters_to_original( optics::get_cluster_indices( wreach, thr ), d.unique_of_original );
	const auto dedup_lbl = labels_from_clusters( n, exp );
	CHECK( rand_index( full_lbl, dedup_lbl ) > 0.85 );
}


TEST_CASE("deduplicate_cosine: collapses same-direction points; weighted sOPTICS agrees (#46)") {
	// Scalar multiples share a direction (cosine-identical) but differ bit-for-bit, so raw
	// deduplicate keeps them apart while deduplicate_cosine (with a small quantum) merges them.
	std::vector<std::array<double, 3>> pts = {
		{ 1, 2, 3 }, { 2, 4, 6 }, { 10, 20, 30 },  // one direction (x3)
		{ 1, 0, 0 }, { 5, 0, 0 },                  // one direction (x2)
		{ 0, 1, 0 } };                             // distinct
	const auto dc = optics::deduplicate_cosine( pts, 1e-6 );
	CHECK( dc.unique_points.size() == 3 );
	std::size_t sumw = 0;
	for ( const auto w : dc.weights ) { sumw += w; }
	CHECK( sumw == pts.size() );
	CHECK( optics::deduplicate( pts ).unique_points.size() == 6 );  // raw dedup keeps all 6

	// End-to-end: weighted sOPTICS on the cosine-deduped cloud agrees with full sOPTICS.
	const auto blobs = optics::testdata::make_blobs<double, 3>( 4, 60, 30.0, 1.0, 808u );
	std::vector<std::array<double, 3>> full;
	std::mt19937 rng( 11 );
	for ( const auto& p : blobs ) {  // replicate each point at several brightnesses (same direction)
		const int reps = 1 + static_cast<int>( rng() % 3 );
		for ( int r = 0; r < reps; ++r ) {
			const double s = 1.0 + 0.5 * static_cast<double>( r );
			full.push_back( { p[0] * s, p[1] * s, p[2] * s } );
		}
	}
	const std::size_t n = full.size();
	const std::size_t mp = 6;
	const double eps = 0.3;
	const auto a = optics::compute_soptics_reachability_dists( full, mp, eps, 256u, 16u, std::size_t{ 32 }, 7u );
	const auto d2 = optics::deduplicate_cosine( full, 1e-4 );
	CHECK( d2.unique_points.size() < full.size() );  // brightness variants collapsed
	const auto wr = optics::compute_soptics_reachability_dists(
		d2.unique_points, mp, eps, 256u, 16u, std::size_t{ 32 }, 7u, 0u, optics::Metric::Cosine, 0.0, d2.weights );
	const double thr = 0.5 * eps;
	const auto la = labels_from_clusters( n, optics::get_cluster_indices( a, thr ) );
	const auto exp2 = optics::expand_clusters_to_original( optics::get_cluster_indices( wr, thr ), d2.unique_of_original );
	const auto lb = labels_from_clusters( n, exp2 );
	CHECK( rand_index( la, lb ) > 0.8 );
}


TEST_CASE("fast Walsh-Hadamard transform (#58 structured projections)") {
	using optics::detail::fwht_inplace;
	using optics::detail::next_pow2;
	using optics::detail::is_pow2;

	CHECK( next_pow2( 0 ) == 1 );
	CHECK( next_pow2( 1 ) == 1 );
	CHECK( next_pow2( 16 ) == 16 );
	CHECK( next_pow2( 17 ) == 32 );
	CHECK( is_pow2( 16 ) );
	CHECK_FALSE( is_pow2( 17 ) );
	CHECK_FALSE( is_pow2( 0 ) );

	// Known transforms (Hadamard-ordered, unnormalized).
	std::vector<double> a = { 1, 0, 0, 0 };
	fwht_inplace( a );
	for ( double v : a ) { CHECK( v == doctest::Approx( 1.0 ) ); }  // delta -> all ones

	std::vector<double> b = { 1, 1, 1, 1 };
	fwht_inplace( b );
	CHECK( b[0] == doctest::Approx( 4.0 ) );
	CHECK( b[1] == doctest::Approx( 0.0 ) );
	CHECK( b[2] == doctest::Approx( 0.0 ) );
	CHECK( b[3] == doctest::Approx( 0.0 ) );

	// Involution up to scale: applying twice multiplies by n.
	std::mt19937 rng( 5 );
	std::normal_distribution<double> g( 0.0, 1.0 );
	const std::size_t n = 8;
	std::vector<double> x( n ), orig( n );
	for ( std::size_t i = 0; i < n; ++i ) { x[i] = g( rng ); orig[i] = x[i]; }
	fwht_inplace( x );
	fwht_inplace( x );
	for ( std::size_t i = 0; i < n; ++i ) { CHECK( x[i] == doctest::Approx( static_cast<double>( n ) * orig[i] ) ); }
}


#ifdef OPTICS_ENABLE_BOOST_RTREE
// Only built when the optional Boost backend is enabled. Verifies that the Boost
// R*-tree backend is interchangeable with nanoflann (issue #27): identical
// neighbor sets at several radii, identical OPTICS ordering end-to-end, and the
// expected dense clusters.
TEST_CASE("boost_backend_tests: nanoflann/boost equivalence") {
	static const int N = 2;
	typedef std::array<double, N> point;
	const std::vector<point> centers = { { 0, 0 }, { 60, 0 }, { 30, 50 } };
	const auto points = optics::testdata::gaussian_blobs<double, N>( centers, 150, 1.5 );

	const optics::NanoflannBackend<double, N> nano( points );
	const optics::BoostRTreeBackend<double, N> boost_be( points );

	// 1) Identical neighbor sets at several radii (sparse, mid, and dense).
	for ( const double eps : { 2.0, 5.0, 12.0 } ) {
		for ( std::size_t i = 0; i < points.size(); ++i ) {
			std::vector<std::size_t> a, b;
			nano.radius_search( points[i], static_cast<double>( eps ), a );
			boost_be.radius_search( points[i], static_cast<double>( eps ), b );
			CHECK( ( sorted( a ) == sorted( b ) ) );
		}
	}

	// 2) Identical OPTICS ordering end-to-end. Equal neighbor sets imply equal
	//    reachability and ordering: tie-breaks are by point index, so neighbor
	//    iteration order (which differs between the backends) does not matter.
	const auto reach_nano = optics::compute_reachability_dists<double, N>( points, 10, 10.0 );
	const auto reach_boost = optics::compute_reachability_dists<double, N, optics::BoostRTreeBackend<double, N>>( points, 10, 10.0 );
	CHECK( ( reach_nano == reach_boost ) );

	// 3) The expected three dense clusters survive a threshold cut.
	const auto clusters = optics::get_cluster_indices( reach_boost, 10.0 );
	std::size_t large = 0;
	for ( const auto& c : clusters ) { if ( c.size() >= 50 ) ++large; }
	CHECK( large == 3 );

	std::cout << "Boost backend equivalence tests successful!" << std::endl;
}
#endif
