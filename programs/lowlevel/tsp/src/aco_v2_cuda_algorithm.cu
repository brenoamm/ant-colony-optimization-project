#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <malloc.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <chrono>

#define CUDA_ERROR_CHECK

#include "../include/aco_v2_cuda_algorithm.cuh"

#include "Randoms.cpp"

#define PHERINIT 0.005
#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 2
#define TAUMAX 2
#define BLOCK_SIZE 32
#define IROULETE 32

__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA ;
__device__ double d_TAUMAX;
__device__ int d_BLOCK_SIZE;
__device__ int d_GRAPH_SIZE;


int NBLOCKS = 0;

std::string::size_type sz;

Randoms *randoms;

// nvcc -o a.out acoCuda.cu
// ./a.out 38 50 50 3 0

using namespace std;

void readMap(double* coord, double* phero, double* dist, int n_cities, int problem){ // small

	printf(" Reading map: %d \n", problem);

	std::ifstream lerMapa;

//	    std::string dji = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/djibouti.txt";
//	    std::string lux = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/luxembourg.txt";
//	    std::string cat = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/catar.txt";
//	    std::string a280 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/a280.txt";
//	    std::string d198 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/d198.txt";
//	    std::string d1291 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/d1291.txt";
//	    std::string lin318 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/lin318.txt";
//	    std::string pcb442 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/pcb442.txt";
//	    std::string pcb1173 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/pbc1173.txt";
//	    std::string pr1002 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/pr1002.txt";
//	    std::string pr2392 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/pr2392.txt";
//	    std::string rat783 = "/home/bamm/Documents/MUESLI/ACO/ACO-low-level/build/Debug/rat783.txt";

		std::string dji = "/home/b/b_mene01/tsp/djibouti.txt";
		std::string lux = "/home/b/b_mene01/tsp/luxembourg.txt";
		std::string cat = "/home/b/b_mene01/tsp/catar.txt";
		std::string a280 = "/home/b/b_mene01/tsp/a280.txt";
		std::string d198 = "/home/b/b_mene01/tsp/d198.txt";
		std::string d1291 = "/home/b/b_mene01/tsp/d1291.txt";
		std::string lin318 = "/home/b/b_mene01/tsp/lin318.txt";
		std::string pcb442 = "/home/b/b_mene01/tsp/pcb442.txt";
		std::string pcb1173 = "/home/b/b_mene01/tsp/pbc1173.txt";
		std::string pr1002 = "/home/b/b_mene01/tsp/pr1002.txt";
		std::string pr2392 = "/home/b/b_mene01/tsp/pr2392.txt";
		std::string rat783 = "/home/b/b_mene01/tsp/rat783.txt";

	    switch (problem) {
			case 1:
				lerMapa.open(dji, std::ifstream::in);
				break;
			case 2:
				lerMapa.open(lux, std::ifstream::in);
				break;
			case 3:
				lerMapa.open(cat, std::ifstream::in);
				break;
			case 4:
				lerMapa.open(a280, std::ifstream::in);
				break;
			case 5:
				lerMapa.open(d198, std::ifstream::in);
				break;
			case 6:
				lerMapa.open(d1291, std::ifstream::in);
				break;
			case 7:
				lerMapa.open(lin318, std::ifstream::in);
				break;
			case 8:
				lerMapa.open(pcb442, std::ifstream::in);
				break;
			case 9:
				lerMapa.open(pcb1173, std::ifstream::in);
				break;
			case 10:
				lerMapa.open(pr1002, std::ifstream::in);
				break;
			case 11:
				lerMapa.open(pr2392, std::ifstream::in);
				break;
			case 12:
				lerMapa.open(rat783, std::ifstream::in);
				break;
		}

    if (lerMapa.is_open()) {

		double randn = 0.0;

		for(int j = 0;j<n_cities;j++){
			for(int k = 0;k<n_cities;k++){
				if(j!=k){
					randn = randoms -> Uniforme() * TAUMAX;
					phero[(j*n_cities) + k] = randn;
					phero[(k*n_cities) + j] = randn;
				}
				else{
					phero[(j*n_cities) + k] = 0;
					phero[(k*n_cities) + j] = 0;
				}
			}
		}

		int i = 0;

		double index, x, y;

		index = 0.0;
				x = 0.0;
				y = 0.0;

		while(!lerMapa.eof()){

			lerMapa >> index;
			lerMapa >> x;
			lerMapa >> y;

			coord[(i*2)] = (double)x;
			coord[(i*2) + 1] = (double)y;

			i+=1;
		}

    }    else{
    	printf(" File not opened\n");
    }
    lerMapa.close();
}


__global__ void setup_rand_kernel(curandState * state, unsigned long seed) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(seed, id, 0, &state[id]);
//	curand_init(1234, id, 0, &state[id]);

  if(id == 0){
	  d_PHERINIT = 0.005;
	  d_EVAPORATION = 0.5;
	  d_ALPHA = 1;
	  d_BETA = 2;
	  d_TAUMAX = 2;
  }

  __syncthreads();
}

__global__ void calculate_distance_kernel(double* dist, double* coord, int n_cities){

    int c_index = threadIdx.x;

    for(int j = 0 ;j<n_cities;j++){
         if(c_index!=j){
            dist[(c_index*n_cities) + j] = sqrt(pow(coord[j*2] - coord[c_index*2],2) + pow(coord[(j*2) + 1] - coord[(c_index*2) + 1],2));
            dist[(j*n_cities) + c_index] = dist[(c_index*n_cities) + j];
         }else{
        	dist[(c_index*n_cities) + j] = 0.0;
         }
    }


}

__global__ void calculate_iroulette_kernel(double* dist, double* coord, int* iroulette, int n_cities){

	int c_index = threadIdx.x;

    //Get the 32 closest nodes for each node.
    for(int i = 0 ; i < IROULETE ; i++){

    	double distance = 999999.9;
    	double c_dist = 0.0;
    	int city = -1;

		for(int j = 0 ;j<n_cities;j++){

			bool check = true;

			for(int k = 0 ; k < i ; k++){
				if(iroulette[c_index * IROULETE + k] == j){
					check = false;
				}
			}

			if(c_index!=j && check){
				c_dist = dist[(c_index*n_cities) + j];
				if(c_dist < distance){
					distance = c_dist;
					city = j;
				}
			}
		}
		iroulette[c_index * IROULETE + i] = city;
	}
}

__global__ void route_kernel(int n_cities, int* routes, double* c_phero, double* c_dist, double* d_probs, curandState* rand_states, double* d_eta, double* d_tau, double* d_sum){

	int ant_index = blockIdx.x;
	int dim_index = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int initialCity = 0;

	//set Initial City for each Ant
	if(dim_index == 0){
		routes[ant_index * n_cities] = initialCity;
	}

	//synchronize
	__syncthreads();

	//start route steps
	for (int i=0; i < n_cities-1; i++) {

		int cityi = routes[ant_index*n_cities+i];

		if(dim_index == 0){
			d_sum[ant_index] = 0.0;
		}

		//synchronize
		__syncthreads();

		d_eta[index] = 0.0;
		d_tau[index] = 0.0;

		if (cityi != dim_index && !vizited(ant_index, dim_index, routes, n_cities, i)){
			d_eta[index] = (double) pow (1 / c_dist[cityi*n_cities+dim_index], d_BETA);
			d_tau[index] = (double) pow (c_phero[(cityi*n_cities)+dim_index],   d_ALPHA);
		}

		if(dim_index == 0){
			for(int j = 0 ; j<n_cities ; j++){
				d_sum[ant_index] += d_eta[ant_index*n_cities+j] * d_tau[ant_index*n_cities+j];
			}
		}

		//synchronize
		__syncthreads();


		//calculate probability to go to city J
		int cityj = dim_index;

		if (cityi == cityj || vizited(ant_index, cityj, routes, n_cities, i)) {
			d_probs[index] = 0;
		}else{
			d_probs[index] = d_eta[index] * d_tau[index] / d_sum[ant_index];
		}

		//choose next city
		if(dim_index == 0){
			int nextCity = city(ant_index, n_cities, d_probs, rand_states);
			routes[(ant_index * n_cities) + (i + 1)] = nextCity;
		}

		//synchronize
		__syncthreads();
	}
}

__global__ void route_kernel2(int n_cities, int* routes, double* c_phero, double* c_dist, double* d_probs,  int* iroulette, curandState* rand_states, double* d_eta, double* d_tau,  double* d_sum){

	int ant_index = blockIdx.x;
	int dim_index = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int next_city = -1;

	routes[ant_index * n_cities] = 0;

	d_sum[ant_index] = 0.0;
	d_eta[index] = 0.0;
	d_tau[index] = 0.0;
	d_probs[index] = 0.0;

	//Loop to build complete route
	for (int i=0; i < n_cities-1; i++) {

		int cityi = routes[ant_index*n_cities+i];

		next_city =  iroulette[(cityi * IROULETE) + dim_index];

		if (cityi != next_city && !vizited(ant_index, next_city, routes, n_cities, i)){
			d_eta[index] = (double) pow (1 / c_dist[cityi*n_cities+next_city], d_BETA);
			d_tau[index] = (double) pow (c_phero[(cityi*n_cities)+next_city], d_ALPHA);
		}

		//synchronize
		__syncthreads();

		if(dim_index == 0){
			for(int j = 0 ; j < IROULETE ; j++){
				d_sum[ant_index] += d_eta[(ant_index*IROULETE)+j] * d_tau[(ant_index*IROULETE)+j];
			}
		}

		//synchronize
		__syncthreads();

		if (cityi == next_city || vizited(ant_index, next_city, routes, n_cities, i)) {
			d_probs[index] = 0;
		}else{
			d_probs[index] = d_eta[index] * d_tau[index] / d_sum[ant_index];
		}

		//choose next city
		if(dim_index == 0){
			if(d_sum[ant_index] > 0.0){
				int nextCity = city(ant_index, n_cities, d_probs, rand_states);
				routes[(ant_index * n_cities) + (i + 1)] = iroulette[cityi*IROULETE+nextCity];
			}else{
				int nc;
				for(nc = 0; nc < n_cities; nc++){
					if(!vizited(ant_index, nc, routes, n_cities, i)){
						break;
					}
				}
				routes[(ant_index * n_cities) + (i + 1)] = nc;
			}

			//clean for next iteration
			d_sum[ant_index] = 0.0;
		}

		d_eta[index] = 0.0;
		d_tau[index] = 0.0;

		//synchronize
		__syncthreads();
	}
}

__global__ void update_pheromones_kernel(int* NUMBEROFANTS, int* NUMBEROFCITIES, int* ROUTES, double* c_phero, double* DELTAPHEROMONES, double* DIST, double* routes_distance, double* bestRoute, int* d_best_sequence) {

//	printf("\n\n\n updatePHEROMONES: ");

	int Q = 11340;
	double RO = 0.5;

	for (int k=0; k<NUMBEROFANTS[0]; k++) {

//		printf("\n N CIties : %i", NUMBEROFCITIES[0]);

		double rlength = d_length(k, NUMBEROFCITIES[0], ROUTES, DIST);
		routes_distance[k] = rlength;

//		printf("\n Distances : %f", rlength);

		for (int r=0; r < NUMBEROFCITIES[0]-1; r++) {

			int cityi = ROUTES[k * NUMBEROFCITIES[0] + r];
			int cityj = ROUTES[k * NUMBEROFCITIES[0] + r + 1];

			DELTAPHEROMONES[cityi* NUMBEROFCITIES[0] + cityj] += Q / rlength;
			DELTAPHEROMONES[cityj* NUMBEROFCITIES[0] + cityi] += Q / rlength;
		}

		if(routes_distance[k] < bestRoute[0]){
			bestRoute[0] = routes_distance[k];
			for (int count=0; count < NUMBEROFCITIES[0]; count++) {
				d_best_sequence[count] = ROUTES[k * NUMBEROFCITIES[0]+count];
			}
		}
	}

	for (int i=0; i<NUMBEROFCITIES[0]; i++) {
		for (int j=0; j<NUMBEROFCITIES[0]; j++) {
			c_phero[i * NUMBEROFCITIES[0] + j] = (1 - RO) * c_phero[i * NUMBEROFCITIES[0] +j] + DELTAPHEROMONES[i * NUMBEROFCITIES[0] +j];
			DELTAPHEROMONES[i * NUMBEROFCITIES[0] +j] = 0.0;

			c_phero[j * NUMBEROFCITIES[0] + i] = (1 - RO) * c_phero[j * NUMBEROFCITIES[0] +i] + DELTAPHEROMONES[j * NUMBEROFCITIES[0] +i];
			DELTAPHEROMONES[j * NUMBEROFCITIES[0] +i] = 0.0;
		}
	}

	__syncthreads();
}

__device__ bool vizited(int antk, int c, int* ROUTES, int NUMBEROFCITIES, int step) {

	for (int l=0; l <= step; l++) {
		if (ROUTES[antk*NUMBEROFCITIES+l] == c) {
			return true;
		}
	}
	return false;
}

__device__ double PHI (int cityi, int cityj, int NUMBEROFCITIES, double* c_dist, double* c_phero, double sum) {

	double dista = c_dist[cityi*NUMBEROFCITIES+cityj];

	double ETAij = (double) pow (1 / dista , d_BETA);
	double TAUij = (double) pow (c_phero[(cityi * NUMBEROFCITIES) + cityj],   d_ALPHA);

	return (ETAij * TAUij) / sum;
}

__device__ int city(int antK, int NCITIES, double* PROBS, curandState* rand_states) {

    double random = curand_uniform(&rand_states[antK]);

	int i = 0;

	double sum = PROBS[antK*IROULETE];
	while (sum < random){
		i++;
		sum += PROBS[antK*IROULETE+i];
	}

	return (int) i;
}

__device__ double d_length (int antk, int NUMBEROFCITIES, int* ROUTES, double* DIST) {

	double sum = 0.0;

	for (int j=0; j<NUMBEROFCITIES-1; j++) {

		int cityi = ROUTES[antk*NUMBEROFCITIES+j];
		int cityj = ROUTES[antk*NUMBEROFCITIES+j+1];

		sum += DIST[cityi*NUMBEROFCITIES + cityj];
	}

	int cityi = ROUTES[antk*NUMBEROFCITIES+NUMBEROFCITIES-1];
	int cityj = ROUTES[antk*NUMBEROFCITIES];

	sum += DIST[cityi*NUMBEROFCITIES + cityj];

	return sum;
}

double run_aco(int n_ant, int n_iterations, int problem){

	int n_cities = 0;
	int n_ants = n_ant;

	NBLOCKS = n_ants;

	switch (problem) {
			case 1:
				n_cities = 38; //Djbouti
				break;
			case 2:
				n_cities = 980; //Luxemburg
				break;
			case 3:
				n_cities = 194; //Catar
				break;
			case 4:
				n_cities = 280;
				break;
			case 5:
				n_cities = 198;
				break;
			case 6:
				n_cities =  1291;
				break;
			case 7:
				n_cities = 318;
				break;
			case 8:
				n_cities = 442;
				break;
			case 9:
				n_cities = 1173;
				break;
			case 10:
				n_cities = 1002;
				break;
			case 11:
				n_cities = 2392;
				break;
			case 12:
				n_cities =  783;
				break;
		}

	randoms = new Randoms(15);

	int GPU_N;
	const int MAX_GPU_COUNT = 1;

	cudaGetDeviceCount(&GPU_N);

	if (GPU_N > MAX_GPU_COUNT) {
		GPU_N = MAX_GPU_COUNT;
	}

	// create stream array - create one stream per GPU
	cudaStream_t stream[GPU_N];

	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(i);
		cudaStreamCreate(&stream[i]);
	}

	//device variables
    double* d_coord;
    double* d_phero;
    double* d_delta_phero;
    double* d_dist;
    double* d_routes_distance;
    double* d_bestRoute;
    double* d_probs;
    double* d_eta;
    double* d_tau;
    double* d_sum;

    int* d_iroulette;
    int* d_seq;
    int* d_nants;
    int* d_ncities;
    int* d_best_sequence;


//    printf("\n Alloc vars \n");
    //Init Random Generators
    curandState* d_rand_states_ind;
	cudaMalloc((void**)&d_rand_states_ind, n_ants * n_cities * sizeof(curandState));

    //alloc host variables
	double* coord = new double[n_cities*2];
	double* phero = new double[n_cities*n_cities];
	double* dist  = new double[n_cities*n_cities];
	int* best_sequence  = new int[n_cities];

//    printf("\n Alloc vars 2 \n");
    //alloc device variables
    cudaMalloc((void**) &d_coord, n_cities*2*sizeof(double));
    cudaMalloc((void**) &d_phero, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_delta_phero, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_dist, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_probs, n_ants*IROULETE*sizeof(double));
    cudaMalloc((void**) &d_routes_distance, n_ants*n_cities*sizeof(double));
    cudaMalloc((void**) &d_bestRoute, sizeof(double));

    cudaMalloc((void**) &d_nants, sizeof(int));
    cudaMalloc((void**) &d_ncities, sizeof(int));

    cudaMalloc((void**) &d_best_sequence, n_cities*sizeof(int));
    cudaMalloc((void**) &d_seq, n_ants*n_cities*sizeof(int));

    cudaMalloc((void**) &d_sum, n_ants*sizeof(double));

    cudaMalloc((void**) &d_iroulette, n_cities*IROULETE*sizeof(int));
    cudaMalloc((void**) &d_eta, n_ants*IROULETE*sizeof(double));
    cudaMalloc((void**) &d_tau, n_ants*IROULETE*sizeof(double));

    setup_rand_kernel<<<n_ants, n_cities, 0, stream[0]>>>(d_rand_states_ind, time(NULL));

//    printf("\n Set rand Kernel \n");

	readMap(coord,phero,dist, n_cities, problem);

//	printf("\n Read Map \n");

	double bestRoute = 99999999.9;
	cudaMemcpy(d_bestRoute, &bestRoute, sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_phero, phero, n_cities*n_cities*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nants, &n_ants, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ncities, &n_cities, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coord, coord, n_cities*2*sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	calculate_distance_kernel<<<1, n_cities>>>(d_dist, d_coord, n_cities); // calculates the distances of each city+
	calculate_iroulette_kernel<<<1, n_cities>>>(d_dist, d_coord, d_iroulette, n_cities); // calculates the distances of each city+

	cudaDeviceSynchronize();

	cudaMemcpy(dist, d_dist, (n_cities*n_cities)*sizeof(double), cudaMemcpyDeviceToHost);

	//Execution Time measure
	double mean_times = 0.0;
	int iteration = 0;

	while(iteration < n_iterations){

		auto t_start = std::chrono::high_resolution_clock::now();

		route_kernel2<<<n_ants, IROULETE>>>(n_cities, d_seq, d_phero, d_dist, d_probs,d_iroulette, d_rand_states_ind, d_eta, d_tau, d_sum);
		cudaDeviceSynchronize();

		auto t_end = std::chrono::high_resolution_clock::now();
		mean_times +=  std::chrono::duration<double>(t_end-t_start).count();

		update_pheromones_kernel<<<1,1>>>(d_nants, d_ncities, d_seq, d_phero, d_delta_phero, d_dist, d_routes_distance, d_bestRoute, d_best_sequence);
		cudaDeviceSynchronize();

		iteration ++;
	}

	printf("\n\n Total time on Tour Construction: %f", mean_times);
	mean_times = mean_times / (n_iterations * n_ants);
	printf("\n\n Average Time on Tour Construction: %f", mean_times);

	cudaDeviceSynchronize();
	cudaMemcpy(best_sequence, d_best_sequence, n_cities*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&bestRoute, d_bestRoute, sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("\n Best PATH %f \n", bestRoute);
	for (int var = 0; var < n_cities; ++var) {
		printf(" %i ", best_sequence[var]);
	}

    return bestRoute;
}
