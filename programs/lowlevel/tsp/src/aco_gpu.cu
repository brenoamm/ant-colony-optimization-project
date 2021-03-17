#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <boost/program_options.hpp>
#include "aco_gpu_algorithm.cuh"

namespace bpo = boost::program_options;

#include <boost/timer/timer.hpp>
namespace bt = boost::timer;

bpo::variables_map g_prog_options;

// ===  FUNCTION  ======================================================================
//         Name:  init_program_options
//  Description: Initializes program options
// =====================================================================================
bool handle_program_options(int argc, char *argv[]) {
  bpo::options_description desc { "Options" };
  desc.add_options()("help,h", "help screen")

  ("runs,r", bpo::value<int>()->default_value(1), "# of runs")

  ("ants,a", bpo::value<int>()->default_value(128), "# of particles")

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


#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

int main(int argc, char **argv)
{
	if (!handle_program_options(argc, argv)) {
	    return 0;
	}

	printf("\n ACO - Begin Of Execution \n");

//	int dim = g_prog_options["dimensions"].as<int>();
	int ants = g_prog_options["ants"].as<int>();
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

	int* best_path = new int[n_cities];
	double* best_length = new double[1];
	double* time_elapsed = new double[1];


	for(int i = 0 ; i < runs ; i++){
		run_aco(ants, iterations, problem, n_cities, best_path, best_length, time_elapsed);

		printf("\n\n\n Run %i Best-Lenght %f Time Elapsed %f", i, best_length[0], time_elapsed[0]);

		printf("\n Best Path: ");

		for(int j = 0 ; j < n_cities ; j++){
			printf(" %i ", best_path[j]);
		}
	}

//	unsigned long long total_time = timer.elapsed().wall;
//	std::cout << "\nExecution time: " << timer.format(5, "%ws") << std::endl;


	printf("\n\n\n End Of Execution");
   return 0;
}
