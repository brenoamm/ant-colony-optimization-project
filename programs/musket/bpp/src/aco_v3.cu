#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include "BPP_0.cu"
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#define CUDA_ERROR_CHECK


// ===  FUNCTION  ======================================================================
//         Name:  main
// =====================================================================================
int main(int argc, char *argv[]){
  char *n_iterationschar = argv[1];
  int iterations = atoi(n_iterationschar);
  char *problemchar = argv[2];
  int problem = atoi(problemchar);
  char *runschar = argv[3];
  int runs = atoi(runschar);

  printf("\n ACO - Begin Of Execution \n");

	//if (!handle_program_options(argc, argv)) {
	//	printf("\n Cant handle anything...");
	//    return 0;
	//}

	//get command line parameters

	printf("\n Starting Execution, Problem %i", problem);
	printf("\n Ants, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Mean Fitness \n");

	int ant[] = {1024, 2048, 4096, 8192};

	for(int ant_setup = 0 ; ant_setup < 4; ant_setup++){
		double mean_fitness = 0.0;

		for(int i = 0 ; i < runs; i++){
			mean_fitness += run_aco(iterations, problem, ant[ant_setup]);
		}

		printf(" %f \n", mean_fitness/runs);
	}
	printf("\n End Of Execution \n");

	 return 0;
}
