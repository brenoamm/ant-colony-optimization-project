#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <climits>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "../include/aco_seq_algorithm.h"

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;


bpo::variables_map g_prog_options;

double actual_execution_time = 0.0;

// ===  FUNCTION  ======================================================================
//         Name:  init_program_options
//  Description: Initializes program options
// =====================================================================================
bool handle_program_options(int argc, char *argv[]) {
   bpo::options_description desc { "Options" };
     desc.add_options()("help,h", "help screen")

     ("runs,r", bpo::value<int>()->default_value(1), "# of runs")

     ("iterations,i", bpo::value<int>()->default_value(5), "# of iterations")

     ("problem,p", bpo::value<int>()->default_value(1), "# problem id");

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

	int iterations = g_prog_options["iterations"].as<int>();
	int problem = g_prog_options["problem"].as<int>();
	int runs = g_prog_options["runs"].as<int>();

	double mean_dist = 0.0;
	double mean_times = 0.0;

	int ant[] = {1024, 2048, 4096, 8192};

	for(int setup = 0 ; setup < 4; setup++){

		mean_dist = 0.0;
		mean_times = 0.0;

		printf("\n\n Starting Test %i with %i ants", setup, ant[setup]);

		for(int i = 0 ; i < runs; i++){

			auto t_start = std::chrono::high_resolution_clock::now();

			mean_dist += init(ant[setup], problem, iterations);

			auto t_end = std::chrono::high_resolution_clock::now();
			mean_times +=  std::chrono::duration<double>(t_end-t_start).count();
		}

		printf("\n\n\n Ending Tests %i with %i ants", setup, ant[setup]);
		printf("\n Mean Time:, %f", mean_times/runs);
	}

	printf("\n End Of Execution \n");

	return 0;
}
