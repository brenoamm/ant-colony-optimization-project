#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include "aco_start.cuh"
#include "../include/musket.cuh"
#include "../include/aco_iroulette_0.cuh"
#define CUDA_ERROR_CHECK

// ===  FUNCTION  ======================================================================
//         Name:  init_program_options
//  Description: Initializes program options
// =====================================================================================


int main(int argc, char *argv[]){

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("\n\n 0000001- cudaCheckError() failed at : %s \n", cudaGetErrorString( err ) );
    }

    char *str_runs = argv[1];
    char *str_iterations = argv[2];
    char *str_problem = argv[3];
    char *str_ants = argv[4];

    int runs =atoi(str_runs);
    int iterations = atoi(str_iterations);
    int problem = atoi(str_problem);
    int ants = atoi(str_ants);

    double mean_dist = 0.0;
    double mean_times = 0.0;

    std::clock_t start;
    start = std::clock();
    double actual_execution_time;

    int ant[] =  {ants};

    mean_dist = 0;
    mean_times = 0.0;

    printf("\n%i;%i;%i;%i;", runs, iterations, problem, ants);

    actual_execution_time = 0.0;
    start = std::clock();
    std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
    mean_dist = run_aco(ant[0], iterations, problem);

    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
    //mean_dist = run_aco(ant[0], iterations, problem);
    
    actual_execution_time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    mean_times = actual_execution_time;
    printf("%0.5f; ", seconds);
      

    return 0;
}
