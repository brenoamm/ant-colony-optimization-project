#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include <boost/program_options.hpp>
#include "mknap_aco_cuda_algorithm.cuh"

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

// ===  FUNCTION  ======================================================================
//         Name:  Main
// =====================================================================================
int main(int argc, char *argv[]){

	printf("\n ACO - Begin Of Execution");

	if (!handle_program_options(argc, argv)) {
		printf("\n Cant handle anything...");
		return 0;
	}

	int runs = g_prog_options["runs"].as<int>();
	int iterations = g_prog_options["iterations"].as<int>();
	int problem = g_prog_options["problem"].as<int>();

	int ant[] = {1024, 2048, 4096, 8192};

	printf("\n run;\titer;\tprob;\tcol size;\tFitness;\tTotal Time ;");

	for(int setup = 0 ; setup < 4; setup++){
		for(int i = 0 ; i < runs; i++){
			printf("\n %i;\t%i;\t%i;\t%i;\t", i, iterations, problem, ant[setup]);

			std::chrono::high_resolution_clock::time_point total_time_start = std::chrono::high_resolution_clock::now();

			double best_fitness = run_aco(ant[setup], iterations, problem);

			std::chrono::high_resolution_clock::time_point total_time_end = std::chrono::high_resolution_clock::now();

			double seconds = std::chrono::duration<double>(total_time_end - total_time_start).count();
			printf("\t%f,\t%f ;", best_fitness, seconds);
		}
	}

	printf("\n End Of Execution \n");
	return 0;
}
