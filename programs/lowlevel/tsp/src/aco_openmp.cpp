#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <climits>
#include <ctime>
#include <chrono>

#include "../include/aco_openmp_algorithm.h"

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;

//#include "file_helper.cpp"

bpo::variables_map g_prog_options;

//#define ITERATIONS		(int) 10
//#define NUMBEROFANTS	(int) 128
//#define NUMBEROFCITIES	(int) 38
//DJIBOUTI 38
//LUXEMBOURG 980

// if (ALPHA == 0) { stochastic search & sub-optimal route }
#define ALPHA			(double) 1
// if (BETA  == 0) { sub-optimal route }
#define BETA			(double) 2
// Estimation of the suspected best route.
#define Q				(double) 11340
// Pheromones evaporation.
#define RO				(double) 0.5
// Maximum pheromone random number.
#define TAUMAX			(int) 2

#define INITIALCITY		(int) 0


// ===  FUNCTION  ======================================================================
// //         Name:  init_program_options
// //  Description: Initializes program options
// // =====================================================================================
bool handle_program_options(int argc, char *argv[]) {
   bpo::options_description desc { "Options" };
     desc.add_options()("help,h", "help screen")

     ("runs,r", bpo::value<int>()->default_value(1), "# of runs")
//
//     ("ants,a", bpo::value<int>()->default_value(2), "# of particles")

     //  ("dimensions,d", bpo::value<int>()->default_value(50), "# of dimensions")

     ("iterations,i", bpo::value<int>()->default_value(5), "# of iterations")

     ("problem,p", bpo::value<int>()->default_value(1), "# problem id");

     //1 - Djbouti , 2 - Luxemburg , 3 - Catar

     //("debug_messages,m", bpo::bool_switch(&debug_msg)->default_value(false),
     //  "print debug messages to console");

     store(parse_command_line(argc, argv, desc), g_prog_options);
     notify(g_prog_options);
     if (g_prog_options.count("help")) {
    	 std::cout << desc << '\n';
         return false;
     }
         return true;
 }

int main(int argc, char **argv) {

	printf("\n Starting Execution");

	if (!handle_program_options(argc, argv)) {
	    return 0;
	}

//	int ants = g_prog_options["ants"].as<int>();
	int iterations = g_prog_options["iterations"].as<int>();
	int problem = g_prog_options["problem"].as<int>();
	int runs = g_prog_options["runs"].as<int>();

	int n_cities = 0;

	switch (problem) {
		case 1:
			n_cities = 38; //Djbouti
			break;
		case 2:
			n_cities = 980; //Luxemburg
			break;
		default:
			n_cities = 194; //Catar
	}

	double mean_dist = 0.0;
	double mean_times = 0.0;

	int ant[] = {1024, 2048, 4096, 8192};

	for(int setup = 0 ; setup < 4; setup++){

		mean_dist = 0.0;
		mean_times = 0.0;

		printf("\n\n\n Starting Test %i with %i ants", setup, ant[setup]);

		for(int i = 0 ; i < runs; i++){

			auto t_start = std::chrono::high_resolution_clock::now();

			mean_dist += run(problem, ant[setup], iterations, runs, 64);

			auto t_end = std::chrono::high_resolution_clock::now();

			mean_times +=  std::chrono::duration<double>(t_end-t_start).count();
		}

	printf("\n Ending Testes %i with %i ants", setup, ant[setup]);
	printf("\n Elapsed Time: %f", mean_times/runs);
	printf("\n Distance: %f", mean_dist/runs);
	}

	return 0;
}
