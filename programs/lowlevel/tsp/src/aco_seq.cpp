#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <climits>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <limits>

#include "../include/aco_seq_algorithm.h"

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;


bpo::variables_map g_prog_options;

double actual_execution_time = 0.0;

// ===  FUNCTION  ======================================================================
//         Name:  init_program_options
//  Description: Initializes program options
// =====================================================================================
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

	double mean_times;

	int ant[] = {1024, 2048, 4096, 8192};
    double max_double = std::numeric_limits<double>::max();
	for(int setup : ant){
		double dist;
		double prev_dist = max_double;
		mean_times = 0.0;

		for(int i = 0 ; i < runs; i++){
			auto t_start = std::chrono::high_resolution_clock::now();
			dist = init(setup, problem, iterations);
            if (dist < prev_dist)
                prev_dist = dist;

			auto t_end = std::chrono::high_resolution_clock::now();
			mean_times += std::chrono::duration<double>(t_end-t_start).count();
		}
        printf("%f;%f;\n", mean_times/runs, prev_dist);
	}
	return 0;
}
