/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "Randoms.cpp"
#include <stdio.h>
#include <stdlib.h>
#include "../include/aco_gpu_algorithm.cuh"
#include <cuda.h>
#include "file_helper.cpp"

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

namespace bt = boost::timer;

#define CUDA_ERROR_CHECK

#define TAUMAX			(int) 2
#define ALPHA			(double) 1
#define BETA			(double) 2
#define Q 				(double) 11340;
#define RO 				(double) 0.5;
#define SEED 			(int)15

//count of GPUS
int GPU_N;
const int MAX_GPU_COUNT = 1;

int n_cities = 0; //Djibouti.txt
//int n_cities = 194; //Catar.txt
//int n_cities = 980; //Luxembourg.txt

Randoms *randoms = new Randoms(15);

//////////////////////////////
//// Define CPU Methods
/////////////////////////////
void connectCities (int* graphs, double* pheromones){
	double randn = 0.0;

	for(int i = 0 ; i<n_cities ; i++){
		for(int j = 0 ; j<n_cities ; j++){
			if(i==j){
				graphs[i*n_cities+j]=0;
				pheromones[i*n_cities+j]=0.0;
			}else{
				randn = randoms -> Uniforme() * TAUMAX;
				graphs[i*n_cities+j]=1;
				pheromones[i*n_cities+j]= randn;
			}
		}
	}
}

//CUDA kernels
__global__ void setup_rand_kernel(curandState * state, unsigned long seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, id, 0, &state[id]);
}

__global__ void distance_kernel(double* cities, double* distance, int* ncities) {

	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = (int)id;

	printf("\n TID : %i", tid);

	double dist = 0.0;

	for(int i = 0 ; i< ncities[0] ; i++){
		dist = sqrt (pow (cities[tid*3+1] - cities[i*3+1], 2) + pow(cities[tid*3+2] - cities[i*3+2], 2));
//		printf(" %f ", dist);
		distance[tid*ncities[0] + i] = dist;
	}
}

__global__ void route_kernel
(int* d_graph, double* d_pheromones, double* d_routes, double* probs,double* d_distances, int* d_n_cities, curandState* rand_states) {

	size_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int sn_cities;

	if(threadIdx.x == 0){
		sn_cities = d_n_cities[0];
	}

	__syncthreads();

	//set Initial city
	d_routes[particle_id*(sn_cities+1)] = 0;

	for (int i=0; i< (sn_cities-1); i++) {

		int cityi = d_routes[(particle_id*(sn_cities+1)+i)];
		int count = 0;

		for (int cityj = 0; cityj < sn_cities; cityj++) {
			//same city
			if (cityi == cityj) {
				continue;
			}
			//if there is connection
			if (exists(cityi, cityj, d_graph, sn_cities)) {

				//if it was not already visited
				if (!vizited (particle_id, cityj, d_routes, sn_cities)) {

					//calculate probabilities
					probs[particle_id*sn_cities*2 + (count * 2)] = PHI(cityi, cityj, particle_id, d_distances, d_pheromones, d_graph, d_routes, sn_cities);
					probs[particle_id*sn_cities*2 + (count * 2) +1] = (double) cityj;

					count++;
				}
			}
		}


		int ncity = city(particle_id, sn_cities, probs, rand_states);


		d_routes[(particle_id * (sn_cities+1)) + i+1] = ncity;

		//restart probabilities
		for(int aba = 0 ; aba < sn_cities; aba++){
			probs[particle_id*sn_cities*2 + aba*2] = -1;
			probs[particle_id*sn_cities*2 + aba*2 +1] = -1;
		}
	}
}

__global__ void length_kernel(int* d_n_cities, double* d_routes,double* d_distances){

	size_t t_id = blockIdx.x * blockDim.x + threadIdx.x;
	int particle_id = (int)t_id;

	length(particle_id, d_n_cities[0], d_routes, d_distances);
}

__global__ void reduction_kernel(int* d_n_particles, int* d_n_cities, double* d_routes, int* d_best_route, double* d_best_route_lenght, double* d_pheromenes, double* d_delta_pheromones){

	int ncities = d_n_cities[0];
	int nparticles = d_n_particles[0];
	double length = 999999999.99;

//	printf("\n\n Number of Cities City : %i", ncities);
//	printf("\n\n Number of particles : %i", nparticles);

	for(int i = 0 ; i < nparticles ; i++){

		length = d_routes[(i*(ncities+1))+ncities];

//		printf("\n\n Particle : %i - Length : %f", i, length);

		if(length < d_best_route_lenght[0]){

			for(int j = 0; j<ncities ; j++){
				d_best_route[j] = (int)d_routes[(i*(ncities+1))+j];
			}
			d_best_route_lenght[0] = length;
		}
	}

	update_Pheromones(nparticles, ncities, d_routes, d_delta_pheromones, d_pheromenes);

	for(int i = 0; i < nparticles ; i++){
		for(int j = 0 ; j < (ncities+1) ; j++){
			d_routes[i*(ncities+1) + j] = -1;
		}
	}

}

//DEVICE Funcitons

__device__ double PHI(int cityi, int c, int id, double* d_distances, double* pheromones, int* graph, double* routes, int n_cities){

	double ETAij = (double) pow (1 / d_distances[cityi*n_cities+c], BETA);
	double TAUij = (double) pow (pheromones[(cityi * n_cities)+ c], ALPHA);

//	printf("\n City1: %i City2: %i" , cityi, c);
//	printf("\n ETA: %f" , ETAij);
//	printf("\n TAUij: %f" , TAUij);

	double sum = 0.0;

	for (int c=0; c < n_cities; c++) {
		if (exists(cityi, c, graph, n_cities)) {
			if (!vizited(id, c, routes, n_cities)) {
				double ETA = (double) pow (1 / d_distances[cityi*n_cities+ c], BETA);
				double TAU = (double) pow (pheromones[(cityi*n_cities)+c], ALPHA);
				sum += ETA * TAU;
			}
		}
	}

//	printf("\n PROB : %f" , (ETAij * TAUij) / sum);
	return (ETAij * TAUij) / sum;
}

__device__ int city(int id, int n_cities, double* probs, curandState * rand_states){

//	printf("\n\n\n CITY -> id : %i, n_cities : %i", id ,n_cities);
//	printf("\n Probs");

//	for(int a = 0 ; a<n_cities-1; a++){
//		printf("\n Probs %f", probs[((id*n_cities*2)+a*2)]);
//	}


	double xi = curand_uniform(&rand_states[id]);
//	printf("\n\n\n XI : %f", xi);

	int index = 0;
//	printf("\n I : %i \n", index);

	double sum = probs[(id*n_cities*2)+index*2];
//	printf("\n SUm : %f", sum);

	while (sum < xi) {
//		printf("\n I : %i", index);
		index++;
		sum += probs[(id*n_cities*2)+(index*2)];
//		printf("\n SUm : %f", sum);
	}

	return (int) probs[(id*n_cities*2)+(index*2)+1];
}

__device__ bool exists(int a, int b, int* graph, int n_cities){
//	if(graph[a*n_cities[0] + b] < 0){
//		return false;
//	}
	return true;
}

__device__ bool vizited(int id, int c, double* routes, int n_cities){

	for(int i = 0 ; i<n_cities ; i++){

		int city = routes[id*(n_cities+i)];
//
//		if(city < 0){
//			return false;
//		}

		if(city == c){
			return true;
		}
	}

	return false;
}

__device__ void length (int ant, int nccities, double* routes, double* distances) {

	double sum = 0.0;
	int cityi = 0;
	int cityj = 0;

	for (int j=0; j< (nccities-1); j++) {
		cityi = routes[ant*(nccities+1)+j];
		cityj = routes[ant*(nccities+1)+j+1];
		sum += distances[cityi*nccities+cityj];
	}

//	printf("\n Distance: %f" , sum);
	routes[ant*(nccities+1)+(nccities)] = sum;
}

__device__ void update_Pheromones(int n_particles, int Ncities, double* routes, double* delta_pheromones, double* pheromones) {
	double q = 11340;
	double ro = 0.5;

	for (int k=0; k<n_particles; k++) {

		double rlength = routes[(k*(Ncities+1)) + (Ncities)];
//		printf("\n Length %f" , rlength);

		for (int r=0; r < Ncities-1; r++) {

			int cityi = routes[k * (Ncities+1) + r];
			int cityj = routes[k * (Ncities+1) + r + 1];

//			printf("\\nn City i %i" , cityi);
//			printf("\n City j %i" , cityj);


			delta_pheromones[(cityi* (Ncities) + cityj)] += (q / rlength);
			delta_pheromones[(cityj* (Ncities) + cityi)] += (q / rlength);
		}
	}

	for (int i=0; i<Ncities; i++) {
		for (int j=0; j<Ncities; j++) {

//			printf("\n\n1 - ro %f", (1 - ro));
//			printf("\nOld pheromone %f", PHEROMONES[i * NUMBEROFCITIES +j]);
//			printf("\ndelta pheromone %f", DELTAPHEROMONES[i * NUMBEROFCITIES +j]);

			pheromones[i * Ncities + j] = (1 - ro) * pheromones[i * Ncities +j] + delta_pheromones[i * Ncities +j];
			delta_pheromones[i * Ncities +j] = 0.0;
		}
	}
}


//Host method
void run_aco(int number_of_particles, int number_of_iterations, int problemId, int number_of_cities, int* best_path, double* best_lenght, double* time_elapsed) {

	printf("\nStarting GPU ACO");

	n_cities = number_of_cities;

	cudaGetDeviceCount(&GPU_N);

	if (GPU_N > MAX_GPU_COUNT) {
		GPU_N = MAX_GPU_COUNT;
	}

	printf("\n\n CUDA-capable device count: %i", GPU_N);

	// create stream array - create one stream per GPU
	cudaStream_t stream[GPU_N];

	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(i);
		cudaDeviceReset();
		cudaStreamCreate(&stream[i]);
	}

	printf("\n Streams Created");

	/////////////////////////////
	//// Define Host related Variables
	/////////////////////////////

	int size_cities_vector = n_cities * 3;
	int size_graph_matrix = n_cities * n_cities;
	int size_routes = number_of_particles * (n_cities + 1); //+1 -> Total distance
	int size_probs = number_of_particles*n_cities*2;

	double* cities = new double[size_cities_vector];
	double* distances = new double[size_graph_matrix];
	int* graph = new int[size_graph_matrix];
	double* pheromones = new double[size_graph_matrix];
	double* delta_pheromones = new double[size_graph_matrix];
	double* routes = new double[size_routes];
	double* probs = new double[size_probs];

	int* best_route_path = new int[n_cities];
	double* best_route_lenght = new double[1];
	best_route_lenght[0] = INT32_MAX;

	int* particles = new int[1];
	particles[0] = number_of_particles;

	size_t cities_size = size_cities_vector * sizeof(double);
	size_t distances_size = size_graph_matrix * sizeof(double);
	size_t routes_size = number_of_particles * (n_cities+1) * sizeof(double);
	size_t probs_size = size_probs * sizeof(double);

	dim3 n_particles_thread = number_of_particles;
	dim3 n_cities_thread = n_cities;
	dim3 n_blocks_thread = 1;

	dim3 n_particles_thread2 = number_of_particles/16;
	dim3 n_blocks_thread2 = 16;

	printf("\n Starting Variables");

	for(int i=0; i < n_cities; i++) {
		for (int j=0; j<3; j++) {
			cities[3*i+j] = -1.0;
		}
		for (int j=0; j < n_cities; j++) {
			distances[n_cities*i + j] = -1.0;
			graph[n_cities*i + j] = -1.0;
			pheromones[n_cities*i + j] = -1.0;
			delta_pheromones[n_cities*i + j] = 0.0;
		}
	}

	for(int i = 0 ; i < number_of_particles; i++){
		for(int j=0; j < n_cities+1; j++) {
			routes[i*(n_cities+1)+j] = -1.0;
			probs[i*n_cities*2]=-1.0;
			probs[i*n_cities*2+1]=-1.0;
		}
	}

	for(int j=0; j < n_cities; j++) {
		best_route_path[j] = 0;
	}

	//Initialize Cities
	readFile(cities, problemId);
	connectCities(graph, pheromones);

	//test
		for(int i=0; i < n_cities; i++) {
			printf("\n City %i " , i);
				for (int j=0; j<2; j++) {
					printf(" x %f y %f:",  cities[3*i+j+1], cities[3*i+j+2]);
				}

			printf("\n\n Pheromones: " );
				for (int j=0; j < n_cities; j++) {
					printf(" %f ", pheromones[n_cities*i + j]);
				}
			}

	/////////////////////////////
	//// Define Device related Variables
	/////////////////////////////

    double* d_cities = new double[n_cities*3];
    double* d_distances = new double[n_cities*n_cities];
    int* d_graph = new int[size_graph_matrix];
    double* d_pheromones = new double[size_graph_matrix];
    double* d_delta_pheromones = new double[size_graph_matrix];
    double* d_routes = new double[size_routes];

    double* d_probs = new double[size_probs];

    int* d_best_route = new int[n_cities];
    double* d_best_route_lenght = new double[1];

    int* d_n_cities = new int[1];
    d_n_cities[0] = n_cities;

    int* d_n_particles = new int[1];
    d_n_particles[0] = number_of_particles;

    int* _cities = new int[1];
    _cities[0] = n_cities;

    //Init Random Generators
    curandState* d_rand_states_ind;
    curandState* d_rand_states_col;

    cudaMalloc(&d_rand_states_ind, number_of_particles * n_cities * sizeof(curandState));
    cudaMalloc(&d_rand_states_col, number_of_particles * sizeof(curandState));

    /////////////////////////////
	//// Alloc Mem Space on Device
	/////////////////////////////

    setup_rand_kernel<<<n_particles_thread, n_cities_thread, 0, stream[0]>>>(d_rand_states_ind, time(NULL));
    cudaDeviceSynchronize();

   	cudaMalloc(&d_cities, cities_size);

   	cudaMalloc(&d_distances, distances_size);
   	cudaMalloc(&d_pheromones, distances_size);
   	cudaMalloc(&d_delta_pheromones, distances_size);

   	cudaMalloc(&d_probs, probs_size);

   	cudaMalloc(&d_routes, routes_size);

   	cudaMalloc(&d_n_cities, sizeof(int));

   	cudaMalloc(&d_best_route, n_cities * sizeof(int));
   	cudaMalloc(&d_best_route_lenght, sizeof(double));

   	cudaMalloc(&d_n_particles, sizeof(int));

    //Copy data to GPU
    cudaMemcpy(d_cities, cities, cities_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, distances, distances_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, graph, n_cities*n_cities*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pheromones, pheromones, distances_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_pheromones, delta_pheromones, distances_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_routes, routes, routes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_probs, probs, probs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_best_route, best_route_path, n_cities * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_best_route_lenght, best_route_lenght, 1*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_particles, particles, 1*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_n_cities, _cities, 1*sizeof(int), cudaMemcpyHostToDevice);

    //Kernel calculates matrix that contains all distances (calculate only once)
    distance_kernel<<<n_blocks_thread, n_cities_thread, 0, stream[0]>>>(d_cities , d_distances, d_n_cities);
    cudaDeviceSynchronize();

    //Get data back from GPU
	cudaMemcpy(distances, d_distances, sizeof(double)*n_cities*n_cities, cudaMemcpyDeviceToHost);

	printf("\n\n Iterations start");

	bt::cpu_timer timer;
	timer.stop();
	timer.start();

    for (int iterations=0; iterations < number_of_iterations; iterations++) {

    	//Calculate routes and lengts
    	route_kernel<<<n_blocks_thread2, n_particles_thread2, 0, stream[0]>>>(d_graph, d_pheromones, d_routes, d_probs,d_distances ,d_n_cities, d_rand_states_ind);
    	cudaDeviceSynchronize();

    	cudaError err = cudaGetLastError();
		if ( cudaSuccess != err )
		{
			printf("\n\n cudaCheckError() failed at : %s\n", cudaGetErrorString( err ) );
		}

    	length_kernel<<<n_blocks_thread2, n_particles_thread2, 0, stream[0]>>>(d_n_cities, d_routes, d_distances);

    	reduction_kernel<<<1, 1, 0, stream[0]>>>(d_n_particles, d_n_cities, d_routes, d_best_route, d_best_route_lenght, d_pheromones, d_delta_pheromones);

	}

    cudaDeviceSynchronize();

	unsigned long long total_time = timer.elapsed().wall;
	std::cout << "\n Execution time: " << timer.format(5, "%ws") << std::endl;

    cudaMemcpy(best_route_path, d_best_route, sizeof(int)*(n_cities), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_route_lenght, d_best_route_lenght, sizeof(double), cudaMemcpyDeviceToHost);

    //Print Results
//    printf("\n\n\n Best Solution; \n");
    for(int j = 0 ; j < (n_cities) ; j++){
//    	printf(" %i ", best_route_path[j]);
    	best_path[j] = best_route_path[j];
	}
//    printf("\n Length : %f \n", best_route_lenght[0]);
    best_lenght[0] = best_route_lenght[0];
    time_elapsed[0] = total_time;
}


