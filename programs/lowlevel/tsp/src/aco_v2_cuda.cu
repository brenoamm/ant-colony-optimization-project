#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include <boost/program_options.hpp>
#include "aco_v2_cuda_algorithm.cuh"
#include "timer.h"
namespace bpo = boost::program_options;

bpo::variables_map g_prog_options;

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
int main(int argc, char *argv[]){

	//printf("\n ACO - Begin Of Execution");
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

	double actual_execution_time;

	int ant[] = {256, 512, 1024, 2048, 4096, 8192};

	for(int setup : ant){
		mean_dist = 0;
		// printf("\n\n\n Starting Test %i with %i ants", setup, ant[setup]);
		for(int i = 0 ; i < runs; i++){
            auto *timer = new msl::Timer();
			mean_dist += run_aco_v2(setup, iterations, problem);
            actual_execution_time = timer->stop();
		}

		printf("%f;", actual_execution_time);
		printf("%f;\n", mean_dist/runs);
		// printf("\n Ending Testes %i with %i ants problem %i", setup, ant[setup], problem);
	}
   return 0;
}
