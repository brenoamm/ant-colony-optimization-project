#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include <chrono>
#include <boost/program_options.hpp>
#include "aco_v1_cuda_algorithm.cuh"

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

//  ("ants,a", bpo::value<int>()->default_value(128), "# of particles")

  ("iterations,i", bpo::value<int>()->default_value(5), "# of iterations")

  ("problem,p", bpo::value<int>()->default_value(1), "# problem id");

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


	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		printf("\n\n 0000001- cudaCheckError() failed at : %s \n", cudaGetErrorString( err ) );
	}

	printf("\n ACO - Begin Of Execution");

	if (!handle_program_options(argc, argv)) {
		printf("\n Cant handle anything...");
	    return 0;
	}

	int runs = g_prog_options["runs"].as<int>();
	int iterations = g_prog_options["iterations"].as<int>();
	int problem = g_prog_options["problem"].as<int>();

//	double mean_dist = 0.0;
//	double mean_times = 0.0;

//	std::clock_t start;
//	start = std::clock();
//	double actual_execution_time;

	int ant[] = {1024, 2048, 4096, 8192};

	printf("\n run; iterations; problem; colony size; Setup rand Kernel; read map; Distance Kernel; Iroulete ; Route ; Pheromone; Best Check;  Copy Data; Total Time ;");

	for(int setup = 0 ; setup < 4; setup++){

//		mean_dist = 0;
//		mean_times = 0;
//		printf("\n\n\n Starting Test %i with %i ants \n", setup, ant[setup]);

		for(int i = 0 ; i < runs; i++){
			printf("\n %i; %i; %i; %i; ", i, iterations, problem, ant[setup]);

			std::chrono::high_resolution_clock::time_point total_time_start = std::chrono::high_resolution_clock::now();

			double mean_dist = run_aco(ant[setup], iterations, problem);

			std::chrono::high_resolution_clock::time_point total_time_end = std::chrono::high_resolution_clock::now();

			double seconds = std::chrono::duration<double>(total_time_end - total_time_start).count();
			printf(" %f ;", seconds);
		}
	}


	printf("\n End Of Execution \n");

   return 0;
}
