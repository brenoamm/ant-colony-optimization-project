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

#include "../include/aco_cuda_algorithm.cuh"
#include "Randoms.cpp"

//ACO Constants
#define PHERINIT 0.005
#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 2
#define TAUMAX 2
#define block_size 64
#define IROULETE 32

//Device Variables
__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA ;
__device__ double d_TAUMAX;
__device__ int d_block_size;
__device__ int d_GRAPH_SIZE;

int n_blocks =0;

std::string::size_type sz;

Randoms *randoms;

using namespace std;

//Read Map and start graphs
void readMap(double* cities, double* phero, double* dist, int n_cities, int problem){ // small

	//printf(" Reading map: %d \n", problem);

	std::ifstream lerMapa;

	std::string dji = "djibouti.txt";
	std::string lux = "luxembourg.txt";
	std::string cat = "catar.txt";
	std::string a280 = "a280.txt";
	std::string d198 = "d198.txt";
	std::string d1291 = "d1291.txt";
	std::string lin318 = "lin318.txt";
	std::string pcb442 = "pcb442.txt";
	std::string pcb1173 = "pbc1173.txt";
	std::string pr1002 = "pr1002.txt";
	std::string pr2392 = "pr2392.txt";
	std::string rat783 = "rat783.txt";

//	std::string dji = "/home/b/b_mene01/tsp/djibouti.txt";
//	std::string lux = "/home/b/b_mene01/tsp/luxembourg.txt";
//	std::string cat = "/home/b/b_mene01/tsp/catar.txt";
//	std::string a280 = "/home/b/b_mene01/tsp/a280.txt";
//	std::string d198 = "/home/b/b_mene01/tsp/d198.txt";
//	std::string d1291 = "/home/b/b_mene01/tsp/d1291.txt";
//	std::string lin318 = "/home/b/b_mene01/tsp/lin318.txt";
//	std::string pcb442 = "/home/b/b_mene01/tsp/pcb442.txt";
//	std::string pcb1173 = "/home/b/b_mene01/tsp/pbc1173.txt";
//	std::string pr1002 = "/home/b/b_mene01/tsp/pr1002.txt";
//	std::string pr2392 = "/home/b/b_mene01/tsp/pr2392.txt";
//	std::string rat783 = "/home/b/b_mene01/tsp/rat783.txt";

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

		while(!lerMapa.eof() && index <= n_cities){

			lerMapa >> index;
			lerMapa >> x;
			lerMapa >> y;

			cities[(i*2)] = (double)x;
			cities[(i*2) + 1] = (double)y;

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

  if(id == 0){
	  d_PHERINIT = 0.005;
	  d_EVAPORATION = 0.5;
	  d_ALPHA = 1;
	  d_BETA = 2;
	  d_TAUMAX = 2;
  }

  __syncthreads();
}

__global__ void calculate_distance_kernel(double* dist, double* cities, int n_cities){

    int c_index = threadIdx.x;

    for(int j = 0 ;j<n_cities;j++){
         if(c_index!=j){
            dist[(c_index*n_cities) + j] = sqrt(pow(cities[j*2] - cities[c_index*2],2) + pow(cities[(j*2) + 1] - cities[(c_index*2) + 1],2));
            dist[(j*n_cities) + c_index] = dist[(c_index*n_cities) + j];
         }else{
        	dist[(c_index*n_cities) + j] = 0.0;
         }
    }
}

__global__ void calculate_iroulette_kernel(double* dist, double* cities, int* iroulette, int n_cities){

	int c_index = threadIdx.x;

    //Get the 32 closest nodes for each node.
    for(int i = 0 ; i< IROULETE ; i++){

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

__global__ void route_kernel(int n_cities, int* routes, double* c_phero, double* c_dist, double* d_probabilities, curandState* rand_states){

	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;
	int initialCity = 0;

	routes[ant_index * n_cities] = initialCity;

	for (int i=0; i < n_cities-1; i++) {

		int cityi = routes[ant_index*n_cities+i];
		int count = 0;

		double sum = 0.0;

		for (int c=0; c < n_cities; c++) {
			if (cityi != c && !vizited(ant_index, c, routes, n_cities, i)){
				double ETA = (double) pow (1 / c_dist[cityi*n_cities+c], d_BETA);
				double TAU = (double) pow (c_phero[(cityi*n_cities)+c],   d_ALPHA);
				sum += ETA * TAU;
			}
		}

		for (int c = 0; c < n_cities; c++) {

			if (cityi == c || vizited(ant_index, c,routes, n_cities, i)) {
				d_probabilities[ant_index*n_cities+c] = 0;
			}else{
				d_probabilities[ant_index*n_cities+c] = PHI(cityi, c, n_cities,  c_dist, c_phero, sum);
				count++;
			}
		}


		// deadlock --- it reaches a place where there are no further connections
		if (0 == count) {
			return;
		}

		int nextCity = city(ant_index, n_cities, d_probabilities, rand_states);

		routes[(ant_index * n_cities) + (i + 1)] = nextCity;
	}

	__syncthreads();
}

__global__ void route_kernel2(int n_cities, int* routes, double* c_phero, double* c_dist, double* d_probabilities,  int* iroulette, curandState* rand_states){

	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;

	int initialCity = 0;
	double sum = 0.0;

	int next_city = -1;
	double ETA = 0.0;
	double TAU = 0.0;

	routes[ant_index * n_cities] = initialCity;

	//Loop to build complete route
	for (int i=0; i < n_cities-1; i++) {

		int cityi = routes[ant_index*n_cities+i];
		int count = 0;

		//loop to calculate all probabilities for the next step
		for (int c = 0; c < IROULETE; c++) {

			next_city =  iroulette[(cityi * IROULETE) + c];

			if (cityi != next_city && !vizited(ant_index, next_city, routes, n_cities, i)){
				ETA = (double) pow (1 / c_dist[cityi*n_cities+ next_city], d_BETA);
				TAU = (double) pow (c_phero[(cityi*n_cities)+ next_city], d_ALPHA);
				sum += ETA * TAU;
			}
		}

		for (int c = 0; c < IROULETE; c++) {

			next_city =  iroulette[(cityi * IROULETE) + c];

			if (cityi == next_city || vizited(ant_index, next_city,routes, n_cities, i)) {
				d_probabilities[ant_index*n_cities+c] = 0;
			}else{
				d_probabilities[ant_index*n_cities+c] = PHI(cityi, next_city, n_cities,  c_dist, c_phero, sum);
				count++;
			}
		}

		// deadlock --- it reaches a place where there are no further connections
		if (0 == count) {
			int nc;

			for(nc = 0; nc < n_cities; nc++){
				if(!vizited(ant_index, nc, routes, n_cities, i)){
					break;
				}
			}
			routes[(ant_index * n_cities) + (i + 1)] = nc;
		}else{
			int chosen_city = city(ant_index, n_cities, d_probabilities, rand_states);
			routes[(ant_index * n_cities) + (i + 1)] = iroulette[cityi*IROULETE+chosen_city];
		}

		sum = 0.0;
	}
	__syncthreads();
}

__global__ void update_pheromones_kernel(int* NUMBEROFANTS, int* n_cities, int* routes, double* c_phero, double* DELTAPHEROMONES, double* distances, double* routes_distance, double* bestRoute, int* d_best_sequence) {

	int Q = 11340;
	double RO = 0.5;

	for (int k=0; k<NUMBEROFANTS[0]; k++) {

		double rlength = d_length(k, n_cities[0], routes, distances);
		routes_distance[k] = rlength;

		for (int r=0; r < n_cities[0]-1; r++) {

			int cityi = routes[k * n_cities[0] + r];
			int cityj = routes[k * n_cities[0] + r + 1];

			DELTAPHEROMONES[cityi* n_cities[0] + cityj] += Q / rlength;
			DELTAPHEROMONES[cityj* n_cities[0] + cityi] += Q / rlength;
		}

		if(routes_distance[k] < bestRoute[0]){
			bestRoute[0] = routes_distance[k];
			for (int count=0; count < n_cities[0]; count++) {
				d_best_sequence[count] = routes[k * n_cities[0]+count];
			}
		}
	}

	for (int i=0; i<n_cities[0]; i++) {
		for (int j=0; j<n_cities[0]; j++) {
			c_phero[i * n_cities[0] + j] = (1 - RO) * c_phero[i * n_cities[0] +j] + DELTAPHEROMONES[i * n_cities[0] +j];
			DELTAPHEROMONES[i * n_cities[0] +j] = 0.0;

			c_phero[j * n_cities[0] + i] = (1 - RO) * c_phero[j * n_cities[0] +i] + DELTAPHEROMONES[j * n_cities[0] +i];
			DELTAPHEROMONES[j * n_cities[0] +i] = 0.0;
		}
	}

	__syncthreads();
}

__device__ bool vizited(int antk, int c, int* routes, int n_cities, int step) {

	for (int l=0; l <= step; l++) {
		if (routes[antk*n_cities+l] == c) {
			return true;
		}
	}
	return false;
}

__device__ double PHI (int cityi, int cityj, int n_cities, double* c_dist, double* c_phero, double sum) {


	double dista = c_dist[cityi*n_cities+cityj];

	double ETAij = (double) pow (1 / dista , d_BETA);
	double TAUij = (double) pow (c_phero[(cityi * n_cities) + cityj],   d_ALPHA);

	return (ETAij * TAUij) / sum;
}

__device__ int city (int antK, int n_cities, double* probabilities, curandState* rand_states) {

    double random = curand_uniform(&rand_states[antK]);

	int i = 0;

	double sum = probabilities[antK*n_cities];

	while (sum < random){ // && i < n_cities-1) {
		i++;
		sum += probabilities[antK*n_cities+i];
	}

	return (int) i;
}

__device__ double d_length (int antk, int n_cities, int* routes, double* distances) {
	double sum = 0.0;
	for (int j=0; j<n_cities-1; j++) {

		int cityi = routes[antk*n_cities+j];
		int cityj = routes[antk*n_cities+j+1];

		sum += distances[cityi*n_cities + cityj];
	}

	int cityi = routes[antk*n_cities+n_cities-1];
	int cityj = routes[antk*n_cities];

	sum += distances[cityi*n_cities + cityj];

	return sum;
}

double run_aco(int n_ant, int n_iterations, int problem){

	int n_cities = 0;
	int n_ants = n_ant;
	n_blocks = n_ant/block_size;

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

	//Host variables
    double* cities = new double[n_cities*2];
    double* phero = new double[n_cities*n_cities];
    double* dist  = new double[n_cities*n_cities];
	int* best_sequence  = new int[n_cities];

	double bestRoute = 99999999.9;

	//device variables
    double* d_cities;
    double* d_pheromones;
    double* d_delta_phero;
    double* d_distances;
    double* d_routes_distance;
    double* d_bestRoute;
    double* d_probabilities;

    int* d_iroulette;
    int* d_routes;
    int* d_nants;
    int* d_ncities;
    int* d_best_sequence;

    //Init Random Generators
    curandState* d_rand_states_ind;
	cudaMalloc(&d_rand_states_ind, n_ants * n_cities * sizeof(curandState));

    //alloc device variables
    cudaMalloc((void**) &d_cities, n_cities*2*sizeof(double));
    cudaMalloc((void**) &d_pheromones, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_delta_phero, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_distances, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_routes, n_ants*n_cities*sizeof(int));
    cudaMalloc((void**) &d_routes_distance, n_ants*n_cities*sizeof(double));
    cudaMalloc((void**) &d_bestRoute, sizeof(double));
    cudaMalloc((void**) &d_nants, sizeof(int));
    cudaMalloc((void**) &d_ncities, sizeof(int));
    cudaMalloc((void**) &d_best_sequence, n_cities*sizeof(int));
    cudaMalloc((void**) &d_probabilities, n_ants*n_cities*sizeof(double));
    cudaMalloc((void**) &d_iroulette, n_cities*IROULETE*sizeof(int));

    cudaDeviceSynchronize();

    setup_rand_kernel<<<n_ants, n_cities, 0, stream[0]>>>(d_rand_states_ind, time(NULL));

	readMap(cities, phero, dist, n_cities, problem);

	cudaMemcpy(d_bestRoute, &bestRoute, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pheromones, phero, n_cities*n_cities*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nants, &n_ants, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ncities, &n_cities, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cities, cities, n_cities*2*sizeof(double), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	calculate_distance_kernel<<<1, n_cities>>>(d_distances, d_cities, n_cities); // calculates the distances of each city+
	calculate_iroulette_kernel<<<1, n_cities>>>(d_distances, d_cities, d_iroulette, n_cities); // calculates the distances of each city+

	cudaDeviceSynchronize();

	cudaMemcpy(dist, d_distances, (n_cities*n_cities)*sizeof(double), cudaMemcpyDeviceToHost);

	//Execution Time measure
	double mean_times = 0.0;
	int iteration = 0;

	while(iteration < n_iterations){

		//auto t_start = std::chrono::high_resolution_clock::now();

		route_kernel2<<<n_blocks, block_size>>>(n_cities, d_routes, d_pheromones, d_distances, d_probabilities,d_iroulette, d_rand_states_ind);
		//cudaDeviceSynchronize();

		//auto t_end = std::chrono::high_resolution_clock::now();
		//mean_times +=  std::chrono::duration<double>(t_end-t_start).count();

		update_pheromones_kernel<<<1,1>>>(d_nants, d_ncities, d_routes, d_pheromones, d_delta_phero, d_distances, d_routes_distance, d_bestRoute, d_best_sequence);
		//cudaDeviceSynchronize();

		iteration ++;
	}

//	printf("\n\n Total time on Tour Construction: %f", mean_times);
//	mean_times = mean_times / (n_iterations * n_ants);
//	printf("\n\n Average Time on Tour Construction: %f", mean_times);

	cudaMemcpy(best_sequence, d_best_sequence, n_cities*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&bestRoute, d_bestRoute, sizeof(double), cudaMemcpyDeviceToHost);

	//cudaDeviceSynchronize();

	printf("\n Best PATH %f \n" , bestRoute);
	for (int var = 0; var < n_cities; ++var) {
		printf(" %i ", best_sequence[var]);
	}

    //cudaDeviceSynchronize();

    return bestRoute;
}

//err = cudaGetLastError();
//if ( cudaSuccess != err )
//{
//	printf("\n\n 3- cudaCheckError() failed at : %s\n", cudaGetErrorString( err ) );
//}
