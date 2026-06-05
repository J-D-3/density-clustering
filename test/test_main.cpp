// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/OPTICS-Clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"

#include "../include/optics/optics.hpp"
#include "../include/optics/testdata.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
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
		for ( auto& n : root.get_children() ) {
			n.add_child( Node( { 2, idx++ } ) );
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


TEST_CASE("convenience: convert_cloud + cluster_dbscan + extract_xi") {
	// Integer cloud (e.g. color-ish data) -> float, then a one-call DBSCAN cut.
	std::vector<std::array<int, 2>> int_pts = {
		{ 100,100 },{ 102,100 },{ 101,101 }, { 0,0 },{ 1,0 },{ 0,1 }, { -100,-100 },{ -102,-100 },{ -101,-101 } };
	auto cloud = optics::convert_cloud<float>( int_pts );
	auto clusters = optics::cluster_dbscan( cloud, 2, 10.0 );
	CHECK( clusters.size() == 3 );

	// Xi clusters as point-index lists from a cluster-ordering.
	std::vector<optics::reachability_dist> reach_dists = {
		{ 1,10.0 },{ 2,9.0 },{ 3,9.0 },{ 4,5.0 },{ 5,5.49 },{ 6,5.0 },
		{ 7,6.5 },{ 8,3.0 },{ 9,2.9 },{ 10,2.8 },{ 11,10.0 },{ 12,12.0 } };
	auto xi = optics::extract_xi( reach_dists, 0.1, 4 );
	CHECK( xi.size() == 3 );
}


#ifdef OPTICS_ENABLE_BOOST_RTREE
// Only built when the optional Boost backend is enabled. Verifies that the
// Boost R*-tree backend returns the same neighbor sets as nanoflann, and that
// it produces the same number of clusters end-to-end.
TEST_CASE("boost_backend_tests") {
	static const int N = 2;
	typedef std::array<double, N> point;
	const std::vector<point> centers = { { 0, 0 }, { 60, 0 }, { 30, 50 } };
	const auto points = optics::testdata::gaussian_blobs<double, N>( centers, 150, 1.5 );

	const optics::NanoflannBackend<double, N> nano( points );
	const optics::BoostRTreeBackend<double, N> boost_be( points );
	const double eps = 5.0;
	for ( std::size_t i = 0; i < points.size(); ++i ) {
		std::vector<std::size_t> a, b;
		nano.radius_search( points[i], eps, a );
		boost_be.radius_search( points[i], eps, b );
		CHECK( sorted( a ) == sorted( b ) );
	}

	const auto reach_boost = optics::compute_reachability_dists<double, N, optics::BoostRTreeBackend<double, N>>( points, 10, 10.0 );
	const auto clusters = optics::get_cluster_indices( reach_boost, 10.0 );
	std::size_t large = 0;
	for ( const auto& c : clusters ) { if ( c.size() >= 50 ) ++large; }
	CHECK( large == 3 );

	std::cout << "Boost backend equivalence tests successful!" << std::endl;
}
#endif
