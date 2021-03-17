#include <vector>
#include <iostream>
#include <cstdarg>
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>

//host COde

double run_aco_bpp(int NANT, int INTER, int problem, int is_palma, int block_setup);

void readBPPFileProperties(int problem, double* n_objects_type, double* bin_capacity);

void readBPPFile(int problem, double* n_objects_type, double* n_objects_total, double* bin_capacity, double* items, int* quantity);

void initializePheromoneMatrix(int n_objects, double* phero);

void checkError(int n);

//Global kernels

__global__ void setup_rand_kernel(curandState* state, unsigned long seed);

__global__ void packing_kernel(double*	d_n_objects,double* d_n_objects_total, double* d_bins_capacity, int* d_n_ants, double* d_phero, double*  d_bpp_items,
		int*  d_bpp_quantity, int*  d_bpp_quantity_copy, int* d_bins, double*  eta, double*  tau, double*  probs, double* d_fitness, curandState* rand_states);

__global__ void evaporation_kernel(double* d_phero);

__global__ void update_pheromones_kernel(double* d_n_objects_total,double* d_n_objects_types, double* d_bins_capacity, double* d_phero, double* d_bpp_items,int* d_bins,double* d_fitness);

__global__ void update_best_fitness_kernel(double* d_n_objects_total, int* d_bins, double* d_fitness, double* d_best_fitness);

__global__ void seq_update_best_fitness_kernel(int* d_n_ants, double* d_n_objects_total, int* d_bins, double* d_fitness, double* d_best_fitness);

__global__ void optimized_update_best_fitness_kernel(double* d_fitness, double* d_best_fitness);

//Aux functions
__device__ double atomicAdd(double* address, double val);
