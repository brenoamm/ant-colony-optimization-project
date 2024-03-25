#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <climits>
#include <ctime>
#include <chrono>

#include "../include/aco_openmp_v2_algorithm.h"

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
void exitWithUsage() {
    std::cerr
            << "Usage: ./gassimulation_test [-g <nGPUs>] [-n <iterations>] [-i <importFile>] [-e <exportFile>] [-t <threads>] [-c <cities>] [-a <ants>] [-r <runs>]"
            << "Default 1 GPU 1 Iteration No import File No Export File threads omp_get_max_threads cities 10 random generated cities ants 16 runs 1" <<std::endl;
    exit(-1);
}
int getIntArg(char *s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        exitWithUsage();
    }
    return i;
}
int main(int argc, char **argv) {

	printf("\n Starting Execution OpenMp v2");

    int num_total_procs, proc_id;
#ifdef MPI_VERSION
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_total_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
#endif
    int iterations = 1, runs = 1;
    std::string problem;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch (argv[i++][1]) {
            case 'i':
                iterations = getIntArg(argv[i], true);
                break;
            case 'p':
                problem = std::string(argv[i]);
                break;
            case 'r':
                runs = (getIntArg(argv[i]));
                break;
            default:
                exitWithUsage();
        }
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

			mean_dist += run(problem, ant[setup], iterations);

			auto t_end = std::chrono::high_resolution_clock::now();

			mean_times +=  std::chrono::duration<double>(t_end-t_start).count();
		}

	printf("\n Ending Testes %i with %i ants", setup, ant[setup]);
	printf("\n Elapsed Time: %f", mean_times/runs);
	printf("\n Distance: %f", mean_dist/runs);
	}

	return 0;
}
