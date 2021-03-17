#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include <boost/program_options.hpp>
#include "aco_v2_cuda_algorithm.cuh"

namespace bpo = boost::program_options;

bpo::variables_map g_prog_options;

// ===  FUNCTION  ======================================================================
//         Name:  init_program_options
//  Description: Initializes program options
// =====================================================================================
bool handle_program_options(int argc, char *argv[]) {
  bpo::options_description desc { "Options" };
  desc.add_options()("help,h", "help screen")

  ("runs,r", bpo::value<int>()->default_value(1), "# of runs")

//  ("ants,a", bpo::value<int>()->default_value(128), "# of particles")

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

int main(int argc, char *argv[]){

	printf("\n ACO - Begin Of Execution");

	if (!handle_program_options(argc, argv)) {
		printf("\n Cant handle anything...");
	    return 0;
	}

//	printf("\n Reading variables \n");

//	int ants = g_prog_options["ants"].as<int>();
	int runs = g_prog_options["runs"].as<int>();
	int iterations = g_prog_options["iterations"].as<int>();
	int problem = g_prog_options["problem"].as<int>();


	double mean_dist = 0.0;
	double mean_times = 0.0;

	std::clock_t start;
	start = std::clock();
	double actual_execution_time;

	int ant[] = {1024, 2048, 4096, 8192};

	for(int setup = 0 ; setup < 4; setup++){

		mean_dist = 0;
		mean_times = 0;

		printf("\n\n\n Starting Test %i with %i ants", setup, ant[setup]);

		for(int i = 0 ; i < runs; i++){

			actual_execution_time = 0.0;
			start = std::clock();

			mean_dist += run_aco(ant[setup], iterations, problem);

			actual_execution_time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
			mean_times += actual_execution_time;
		}

		printf("\n\n Elapsed Time: %f", mean_times/runs);
		printf("\n Distance: %f", mean_dist/runs);
		printf("\n Ending Testes %i with %i ants problem %i", setup, ant[setup], problem);
	}


	printf("\n End Of Execution");

   return 0;
}
