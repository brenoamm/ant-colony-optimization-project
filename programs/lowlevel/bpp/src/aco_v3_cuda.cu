#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include <boost/program_options.hpp>
#include "aco_v3_cuda_algorithm.cuh"
#include <chrono>

#define CUDA_ERROR_CHECK

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
			("problem,p", bpo::value<int>()->default_value(1), "# problem id")
			("palma,c", bpo::value<int>()->default_value(0), "flag for cluster execution");

	store(parse_command_line(argc, argv, desc), g_prog_options);
	notify(g_prog_options);

	if (g_prog_options.count("help")) {
		std::cout << desc << '\n';
		return false;
	}

	return true;
}

// ===  FUNCTION  ======================================================================
//         Name:  main
// =====================================================================================
int main(int argc, char *argv[]){

//	printf("\n ACO - Begin Of Execution \n");

	if (!handle_program_options(argc, argv)) {
		printf("\n Cant handle anything...");
	    return 0;
	}

	//get command line parameters
	int runs = g_prog_options["runs"].as<int>();
	int iterations = g_prog_options["iterations"].as<int>();
	int problem = g_prog_options["problem"].as<int>();
	int palma_flag = g_prog_options["palma"].as<int>();

//	printf("\n Starting Execution, Problem %i", problem);
	printf("Problem, Block-Size,  Ants, Time,  Fitness \n");

	int block_size[] = {256,512};
	int ant[] = {1024, 2048, 4096, 8192};

	for(int block_setup = 0 ; block_setup < 1; block_setup++){
		for(int ant_setup = 0 ; ant_setup < 4; ant_setup++){
			double mean_fitness = 0.0;

			for(int i = 0 ; i < runs; i++){
				mean_fitness += run_aco_bpp(ant[ant_setup], iterations, problem, palma_flag, block_size[block_setup]);
			}

//			printf(" %f \n", mean_fitness/runs);
		}

	}
	printf("\n End Of Execution \n");

	 return 0;
}
