#include <vector>
#include <iostream>
#include <cstdarg>
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>

//host COde

double run_aco(int NANT, int INTER, int problem);

void readMap(double* coord, double* phero, double* dist, int NCITY, int problem);

//Global kernels

__global__ void setup_rand_kernel(curandState * state, unsigned long seed);

__global__ void calculate_distance_kernel(double* dist, double* coord, int NCITY);

__global__ void calculate_iroulette_kernel(const double* dist, int* iroulette, int NCITY);

__global__ void route_kernel(int NCITY, int* routes, double* c_phero, const double* c_dist, double* d_probs, const int* iroulette, curandState* rand_states, double* d_sum);//double* d_eta, double* d_tau, double* d_sum);

__global__ void evaporation_kernel(double* c_phero);

__global__ void best_ant_kernel(const int NUMBEROFANTS, const int NUMBEROFCITIES, const int* ROUTES, const double* routes_distance, double bestRoute, int* d_best_sequence);

__global__ void calc_route_distance(const int * d_seq, const double * d_dist, double * d_routes_distance, int n_cities);

//Device Functions

__device__ double PHI (int cityi, int cityj, int NCITY, const double* c_dist, double* c_phero, double sum);

__device__ double d_length (int antk, int NUMBEROFCITIES, int* ROUTES, double* DIST);

__device__ bool visited(int antk, int c, const int* ROUTES, int NUMBEROFCITIES, int i);

__device__ int city (int antK, double* PROBS, curandState* rand_states);

