/*
 * aco_gpu_algorithm.cuh
 *
 *  Created on: MAR 5, 2018
 *      Author: Breno Menezes
 */
#pragma once

#include <vector>
#include <iostream>
#include <cstdarg>
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>


// ===  FUNCTION  ======================================================================
//         Name:  run_fss
//  Description:  Runs the FSS algorithm
// =====================================================================================
void run_aco(int number_of_particles, int number_of_iterations, int problemId, int n_cities, int* best_path, double* best_lenght, double* time_elapsed);

void connectCities (int* graphs, double* pheromones);

__device__ double PHI(int cityi, int c, int id, double* d_distances, double* pheromones, int* graph, double* routes, int n_cities);

__device__ bool exists(int a, int b, int* graph, int n_cities);

__device__ bool vizited(int id, int c, double* routes, int n_cities);

__device__ void length (int ant, int nccities, double* routes, double* distances);

__device__ int city(int id, int n_cities, double* probs, curandState * rand_states);

__device__ void update_Pheromones(int NUMBEROFANTS, int NUMBEROFCITIES, double* ROUTES, double* DELTAPHEROMONES, double* PHEROMONES);

__global__ void route_kernel (int* d_graph, double* d_pheromones, double* d_routes, double* probs,double* d_distances, int* d_n_cities, curandState * rand_states);

__global__ void distance_kernel(double* cities, double* distance, int n_cities);

__global__ void length_kernel(int* d_n_cities[0],double* d_routes,double* d_distances);

__global__ void reduction_kernel(int* d_n_particles, int* d_n_cities, double* d_routes, int* d_best_route, double* d_best_route_lenght, double* d_pheromenes, double* d_delta_pheromones);
