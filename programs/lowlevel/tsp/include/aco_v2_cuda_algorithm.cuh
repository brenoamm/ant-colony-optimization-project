#include <vector>
#include <iostream>
#include <cstdarg>
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>

//host COde

double run_aco(int NANT, int INTER, int problem);
double run_aco_v2(int NANT, int INTER, const std::string& problem);

void readMap(double* coord, double* phero, double* dist, int NCITY, int problem);
void readMap(double* coord, double* phero, double* dist, int NCITY, const std::string& problem);

//Global kernels

__global__ void setup_rand_kernel(curandState * state, unsigned long seed);

__global__ void calculate_distance_kernel(double* dist, double* coord, int NCITY);

__global__ void calculate_iroulette_kernel(double* dist, double* coord, int* iroulette, int NCITY);

__global__ void route_kernel(int NCITY, int* ROUTES,  double* PHERO, const double* c_dist, double* d_probs, curandState* rand_states, double* d_eta, double* d_tau,double* d_sum);

__global__ void route_kernel2(int NCITY, int* routes, double* c_phero, const double* c_dist, double* d_probs, const int* iroulette, curandState* rand_states,  double* d_eta, double* d_tau,double* d_sum);

__global__ void update_pheromones_kernel(int* NUMBEROFANTS, int* NUMBEROFCITIES, int* ROUTES,double* c_dist, double* PHEROMONES, double* DELTAPHEROMONES, double* DIST, double* routes_distance, double* bestRoute, int* d_best_sequence);

//__global__ void PHI (int cityi, int cityj, int NUMBEROFCITIES, double* c_dist, double* c_phero, double sum);

//Device Functions

__device__ double d_length (int antk, int NUMBEROFCITIES, const int* ROUTES, const double* DIST);

__device__ bool vizited(int antk, int c, const int* ROUTES, int NUMBEROFCITIES, int i);

//__device__ double PHI (int cityi, int cityj, int NUMBEROFCITIES, double* c_dist, double* c_phero, double sum);

__device__ double PHI (int cityi, int cityj, int NUMBEROFCITIES, const double* c_dist, double* c_phero, double sum);

__device__ int city (int antK, [[maybe_unused]] int NCITIES, const double* PROBS, curandState* rand_states);
