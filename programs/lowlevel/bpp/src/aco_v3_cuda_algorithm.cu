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

#include "../include/aco_v3_cuda_algorithm.cuh"
#include "Randoms.cpp"

#define PHERINIT 0.005
#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 1
#define TAUMAX 2
#define IROULETE 32
#define Q 32

__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA ;
__device__ double d_TAUMAX;
__device__ int d_BLOCK_SIZE;
__device__ int d_GRAPH_SIZE;

std::string::size_type sz;

int BLOCK_SIZE = 256;

Randoms *randoms;

bool is_palma = false;

using namespace std;

// ===  HOST FUNCTIONS =================================================================
// =====================================================================================
// ===  FUNCTION  ======================================================================
//         Name:  readBPPFileProperties
//         Description:  read BPP file containing the information about how many product
//			exist in order to create the right structure for the objects.
// =====================================================================================
void readBPPFileProperties(int problem, double* n_objects_type, double* bin_capacity){
//	printf("\n\n Reading BPP File");

	std::ifstream fileReader;

	//Problem Instances
	std::string palma_path = "";
	if(is_palma){
		palma_path = "/home/b/b_mene01/ls_pi-research_aco-bpp/BPP/LowLevelProgram/build/release/";
	}

	//Problem Instances
	std::string f60 = "Falkenauer_t60_00.txt";
	std::string p201 = "201_2500_NR_0.txt";
	std::string p402 = "402_10000_NR_0.txt";
	std::string p600 = "600_20000_NR_0.txt";
	std::string p801 = "801_40000_NR_0.txt";
	std::string p1002 = "1002_80000_NR_0.txt";

	switch(problem){
	case 0:
		fileReader.open(palma_path+f60, std::ifstream::in);
		break;
	case 1:
		fileReader.open(palma_path+p201, std::ifstream::in);
		break;
	case 2:
		fileReader.open(palma_path+p402, std::ifstream::in);
		break;
	case 3:
		fileReader.open(palma_path+p600, std::ifstream::in);
		break;
	case 4:
		fileReader.open(palma_path+p801, std::ifstream::in);
		break;
	case 5:
		fileReader.open(palma_path+p1002, std::ifstream::in);
		break;
	default:
		break;
	}


	if (fileReader.is_open()) {

		fileReader >> n_objects_type[0];
		fileReader >> bin_capacity[0];
	}
	fileReader.close();
}


// ===  FUNCTION  ======================================================================
//         Name:  readBPPFile
//         Description:  read BPP file containing the problem instance values
// =====================================================================================
void readBPPFile(int problem, double* n_objects_type, double* n_objects_total, double* bin_capacity, double* items, int* m_quantity){

	std::ifstream fileReader;

	std::string palma_path = "";
	if(is_palma){
		palma_path = "/home/b/b_mene01/ls_pi-research_aco-bpp/BPP/LowLevelProgram/build/release/";
	}

	//Problem Instances
	std::string f60 = "Falkenauer_t60_00.txt";
	std::string p201 = "201_2500_NR_0.txt";
	std::string p402 = "402_10000_NR_0.txt";
	std::string p600 = "600_20000_NR_0.txt";
	std::string p801 = "801_40000_NR_0.txt";
	std::string p1002 = "1002_80000_NR_0.txt";

	switch(problem){
	case 0:
		fileReader.open(palma_path+f60, std::ifstream::in);
		break;
	case 1:
		fileReader.open(palma_path+p201, std::ifstream::in);
		break;
	case 2:
		fileReader.open(palma_path+p402, std::ifstream::in);
		break;
	case 3:
		fileReader.open(palma_path+p600, std::ifstream::in);
		break;
	case 4:
		fileReader.open(palma_path+p801, std::ifstream::in);
		break;
	case 5:
		fileReader.open(palma_path+p1002, std::ifstream::in);
		break;
	default:
		break;
	}

	int lines = 0;
	double total = 0.0;

	if (fileReader.is_open()) {

		fileReader >> n_objects_type[0];
		fileReader >> bin_capacity[0];

		while (lines < n_objects_type[0] && !fileReader.eof()) {
			double weight;
			double quantity;

			fileReader >> weight;
			fileReader >> quantity;

			items[lines] = weight;
			m_quantity[lines] = quantity;

			total+=quantity;

			lines++;
		}
	}
	 else{
		 printf("\n File not opened");
	}

	n_objects_total[0] = total;

//	printf("\n Object Types: %f" , n_objects_type[0]);
//	printf("\n Object Total: %f" , n_objects_total[0]);

	fileReader.close();
}

// ===  FUNCTION  ======================================================================
//         Name:  initializePheromoneMatrix
//         Description:  Set Random values to the pheromone matrix
// =====================================================================================
void initializePheromoneMatrix(int n_objects, double* phero){

	double randn = 0.0;

	for(int j = 0;j<n_objects;j++){
		for(int k = 0;k<n_objects;k++){
//			if(j!= k){
				randn = randoms -> Uniforme() * TAUMAX;
				phero[(j*n_objects) + k] = randn;
				phero[(k*n_objects) + j] = randn;
//			}
//			else{
//				phero[(j*n_objects) + k] = 0.0;
//				phero[(k*n_objects) + j] = 0.0;
//			}
		}
	}
}

// ===  CUDA KERNELS ===================================================================
// =====================================================================================
// ===  FUNCTION  ======================================================================
//         Name:  setup_rand_kernel
//         Description:  Initialize CUDA's random generators
// =====================================================================================
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
}

// ===  FUNCTION  ======================================================================
//         Name:  packing_kernel
//         Description:
// =====================================================================================
__global__ void packing_kernel(double*	d_n_objects_types, double* d_n_objects_total,
		double* d_bins_capacity, int* d_n_ants, double* d_phero, double* d_bpp_items,
		int*  d_bpp_quantity, int*  d_bpp_quantity_copy, int* d_bins, double*  d_eta, double*  d_tau, double*  d_probs,
		double* d_fitness, curandState* rand_states){


//	printf("\n n antes %i", d_n_ants[0]);
	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;

	double n_objects_type = d_n_objects_types[0];
	double n_objects_total = d_n_objects_total[0];
	double bin_capacity = d_bins_capacity[0];

	//Actual Solution Index
	int object_bin_index = ant_index * n_objects_total; //counts the position to add new objects to the solutions where each ant has a part sized = total_objects
	int local_type_bin_index = ant_index * n_objects_type; //Counts the position of the parameters with a referent to the items = type_objects
	int bins_used = 0;
	double actual_bin_weight = 0.0;
	int n_items_in_actual_bin = 0;

	//Copy Object and Quantities array
//	for(int i = 0 ; i < (int) n_objects_type ; i++){
//		int index = (((int)ant_index)*(n_objects_type)) + i;
//		d_bpp_quantity_copy[index] = d_bpp_quantity[i];
//	}

	//prefix
	int bpp_items_copy_index = (int)ant_index*n_objects_type;

	//Used to check if there are still objects that could fit in the actual bin
	int possible_items_to_this_bin = 0;

	//Start first bin -> Get heaviest item available and add to first bin
	int object_index = 0;
	double object_weight = 0.0;
	double object_quantity = 0.0;
	double new_object_weight = 0.0;
	double eta_tau_sum = 0.0;

	int index = 0;

	//Get heaviest Object to start Bin
	for(int i = 0 ; i < n_objects_type; i++){
		index = (((int)ant_index)*(n_objects_type)) + i;
		d_bpp_quantity_copy[index] = d_bpp_quantity[i];

		new_object_weight = d_bpp_items[(i)];
		object_quantity = d_bpp_quantity_copy[bpp_items_copy_index + i];

		if((object_quantity > 0) && (new_object_weight > object_weight)){
			object_index = i;
			object_weight = new_object_weight;
		}
	}

	//Add object
	d_bins[object_bin_index] = object_index;
	n_items_in_actual_bin++;
	actual_bin_weight += object_weight;
	bins_used++;

	//Remove from available itens
	d_bpp_quantity_copy[bpp_items_copy_index + (object_index)] -= 1.0;

	//	printf("\n packing Item %i      - Weight %f ", object_index, object_weight);

	//Loop to build complete bins
	for (int i = 0; i < n_objects_total-1; i++) {

//		if((ant_index == 0)){
//			printf("\n new");
//		}

		eta_tau_sum = 0.0;
		possible_items_to_this_bin = 0;

		//Loop to check the possibility of adding other objects
		for (int index_object_j = 0; index_object_j < n_objects_type; index_object_j++) {

			//printf("\n Calc Probabilities");

			d_eta[local_type_bin_index+index_object_j] = 0.0;
			d_tau[local_type_bin_index+index_object_j] = 0.0;
			d_probs[local_type_bin_index+index_object_j] = 0.0;

			//Get data from the object list
			double weight_object_j = d_bpp_items[index_object_j];
			int quantity_object_j = (int)d_bpp_quantity_copy[bpp_items_copy_index + index_object_j];

			//Check if there is still objects available and if the weight suits actual bin
			if((quantity_object_j > 0) && (weight_object_j <= (bin_capacity-actual_bin_weight))){

				//Calculate the first part of the probability calculation
//				if(actual_bin_weight == 0.0){
//					d_eta[local_type_bin_index+index_object_j] = 1.0;
//					printf("\n não entre aqui");
//				}else{
				for(int k = 0 ; k < n_items_in_actual_bin ; k++){
					//last item added to by this ant = index + 1
					//Stay inside last bin using k
					int object_i = d_bins[object_bin_index+i-k];

					d_eta[local_type_bin_index + index_object_j] += d_phero[object_i*(int)n_objects_type + index_object_j];

				}
				d_eta[local_type_bin_index+index_object_j] = d_eta[local_type_bin_index+index_object_j] / n_items_in_actual_bin;
//				}

				//Calculate the second part of the probability calculation
				d_tau[local_type_bin_index + index_object_j] = (double) pow(weight_object_j, BETA);

				eta_tau_sum += d_eta[local_type_bin_index + index_object_j] * d_tau[local_type_bin_index + index_object_j];
				possible_items_to_this_bin++;
			}
		}

//		printf("\n possible entries : %i , etatausum %f", possible_items_to_this_bin, eta_tau_sum);

		if(possible_items_to_this_bin > 0){

			//Loop to Calculate probabilities based on the values calculated above
			for (int index_object_j = 0; index_object_j < n_objects_type; index_object_j++) {

				double p = (d_eta[local_type_bin_index+index_object_j] * d_tau[local_type_bin_index+index_object_j]) / eta_tau_sum;
//				if(isnan(p)){
//					printf("\n NAN ----> %f  |||| %f |||||| %f", d_eta[local_type_bin_index+index_object_j], d_tau[local_type_bin_index+index_object_j], eta_tau_sum);
//				}
				d_probs[local_type_bin_index+index_object_j] = p;
			}


			//Add new object in a probabilistic manner
			double random = curand_uniform(&rand_states[ant_index]);
			int select_index = 0;
			int selected_object = -1;
			double sum = 0.0;
			double prob = 0.0;

			while ((sum <= random) && (select_index < n_objects_type)){

				prob = d_probs[local_type_bin_index+select_index];
				if(prob > 0.0){
					sum += prob;
					selected_object = select_index;
				}

				select_index++;
			}

//			if(selected_object == -1){
//				printf("\n PAM PAM Random %f, SUM %f", random, sum );
//				for (int index_object_j = 0; index_object_j < n_objects_type; index_object_j++) {
//					printf("\n PROB %i - %f ", index_object_j, d_probs[local_type_bin_index+index_object_j]);
//					printf("\n ETA TAU SUM %f ", eta_tau_sum);
//
//				}
//
//			}

			//Add selected object to the list
			d_bins[ant_index*(int)n_objects_total+i+1] = selected_object;

			//Add weight to actual bin
			double weight_object_j = d_bpp_items[selected_object];
			actual_bin_weight += weight_object_j;

//			printf("\n packing Item %i      - Weight %f ", object_j, weight_object_j);

			//Remove one available item + Increase items in Bin + reset number of possible items to the bin.
			d_bpp_quantity_copy[bpp_items_copy_index + selected_object] -= 1.0;
			n_items_in_actual_bin++;


		}else{

//			printf("\n\n New BIN ");
			//Start new BIN
			//Start first bin -> Get heaviest item available and add to first bin
			bins_used++;

			object_index = 0;
			object_weight = 0.0;
			object_quantity = 0.0;
			new_object_weight = 0.0;

			for(int k = 0 ; k < n_objects_type ; k++){

				object_quantity = d_bpp_quantity_copy[bpp_items_copy_index +k];
				new_object_weight = d_bpp_items[k];

				if((object_quantity > 0) && (new_object_weight > object_weight)){
					object_index = k;
					object_weight = new_object_weight;
				}
			}

			d_bpp_quantity_copy[bpp_items_copy_index + object_index]  -= 1.0 ;
			d_bins[ant_index*(int)n_objects_total+i+1] = object_index;

			n_items_in_actual_bin = 1;
			actual_bin_weight = object_weight;
		}
	}

	//set_fitness of current Ant
	d_fitness[ant_index] = 1.0 * bins_used;
}

// ===  FUNCTION  ======================================================================
//         Name:  evaporation_kernel
//         Description:
// =====================================================================================
__global__ void evaporation_kernel(double* d_phero) {

	int x_index = blockIdx.x * blockDim.x + threadIdx.x;

	//Evaporation Rate
	double RO = EVAPORATION;

	if(blockIdx.x !=  threadIdx.x){
		d_phero[x_index] = (1 - RO) * d_phero[x_index];
	}

}

// ===  FUNCTION  ======================================================================
//         Name:  evaporation_kernel
//         Description:
// =====================================================================================
__global__ void update_pheromones_kernel(double* d_n_objects_total, double* d_n_objects_type, double* d_bins_capacity,
		double* d_phero, double* d_bpp_items,int* d_bins, double* d_fitness){

	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;
	int n_objects_total = d_n_objects_total[0];
	int n_objects_types = d_n_objects_type[0];
	double bin_capacity = d_bins_capacity[0];

	//double ant_fitness = d_fitness[ant_index];

	double actual_bin_weight = 0.0;
	int actual_bin_object_index = 0;
	int actual_bin_n_objects = 0;

	for(int i = 0 ; i < n_objects_total ; i++){

		double object_weight = d_bins[ant_index*n_objects_total+i];

		if(actual_bin_weight + object_weight < bin_capacity){
			actual_bin_n_objects++;
			actual_bin_weight+=object_weight;
		}else{
			//update pheromones between items from actual bin index -> n-objects
			for(int j = 0; j<actual_bin_n_objects; j++){
				for(int k = j+1; k<actual_bin_n_objects; k++){

					int object_i = d_bins[ant_index*n_objects_total+actual_bin_object_index+j];
					int object_j = d_bins[ant_index*n_objects_total+actual_bin_object_index+k];

					double delta_pheromone =  Q / d_fitness[ant_index];

					atomicAdd(&d_phero[object_i * n_objects_types + object_j],  delta_pheromone);
					atomicAdd(&d_phero[object_j * n_objects_types + object_i],  delta_pheromone);
				}
			}

			//Start new bin count
			actual_bin_n_objects = 1;
			actual_bin_weight = object_weight;
			actual_bin_object_index = i;
		}
	}
}

__global__ void update_best_fitness_kernel(double* d_n_objects_total, int* d_bins, double* d_fitness, double* d_best_fitness){

	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;

	double ant_fitness = d_fitness[ant_index];

	if(ant_fitness < d_best_fitness[0]){
		d_best_fitness[0] = ant_fitness;
		//printf("\n new best %f.0", ant_fitness);
	}
}

__global__ void seq_update_best_fitness_kernel(int* d_n_ants, double* d_n_objects_total, int* d_bins, double* d_fitness, double* d_best_fitness){

	for(int i = 0 ; i < d_n_ants[0] ; i++){
		double ant_fitness = d_fitness[i];

		if(ant_fitness < d_best_fitness[0]){
			d_best_fitness[0] = ant_fitness;
//			printf("\n new best %f", d_best_fitness[0]);
		}
	}
}

// 1 step parallel reduction like in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// It uses shared memory
__global__ void optimized_update_best_fitness_kernel(double* d_fitness, double* d_best_fitness){
	extern __shared__ int sdata[];

	// each thread loadsone element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i= blockIdx.x*blockDim.x+ threadIdx.x;

	sdata[tid] = d_fitness[i];

	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if(tid % (2*s) == 0){
			if(sdata[tid + s] < sdata[tid]){
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if(tid == 0){
		d_best_fitness[0] = sdata[0];
	}
}


// Starting point
double run_aco_bpp(int n_ant, int n_iterations, int problem_id, int isPalma, int block_setup){

	BLOCK_SIZE = block_setup;

	if(isPalma == 1){
		is_palma = true;
	}

	//Start GPUs
	int GPU_N;
	const int MAX_GPU_COUNT = 1;
	cudaGetDeviceCount(&GPU_N);

	if (GPU_N > MAX_GPU_COUNT) {
		GPU_N = MAX_GPU_COUNT;
	}

	//printf("\n CUDA-capable device count: %i", GPU_N);
	// create stream array - create one stream per GPU
	cudaStream_t stream[GPU_N];

	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(i);
		cudaDeviceReset();
		cudaStreamCreate(&stream[i]);
	}

//	if(n_ant == 8192){
//		BLOCK_SIZE = 256;
//	}

	int n_blocks = n_ant / BLOCK_SIZE;
	int n_threads = n_ant / n_blocks;

//	Create and Allocate device variables
    double* d_phero;
    double* d_delta_phero;
    double* d_fitness;
    double* d_best_fitness;
    double* d_probabilities;
    double* d_n_objects_type;
    double* d_n_objects_total;

    double* d_bin_capacity;
    double* d_eta;
	double* d_tau;
	double* d_sum;

    int* d_bins;
    int* d_n_ants;
    int* d_best_solution;

    double* d_bpp_items;
    int* d_bpp_items_quantity;
    int* d_bpp_items_quantity_copy;

    randoms = new Randoms(15);

    //Initialize Host Structures and Read File in order to allocate device structures
    double* n_objects_types = (double*)malloc(1*sizeof(double));
    double* n_objects_total = (double*)malloc(1*sizeof(double));
    double* bin_capacity = (double*)malloc(1*sizeof(double));

    readBPPFileProperties(problem_id, n_objects_types, bin_capacity);

    double* bpp_items = (double*)malloc(n_objects_types[0]*sizeof(double));
    int* bpp_qty = (int*)malloc(n_objects_types[0]*sizeof(int));

    readBPPFile(problem_id, n_objects_types, n_objects_total, bin_capacity, bpp_items, bpp_qty);

    //Init Random Generators
    curandState* d_rand_states_ind;
	cudaMalloc(&d_rand_states_ind, n_ant * sizeof(curandState));

	//alloc other host variables
	int bin_capacity_size = bin_capacity[0];
	int n_object_type = n_objects_types[0];
	int n_object_total = n_objects_total[0];
	int pheromone_matrix_size = n_objects_types[0] * n_objects_types[0];

	double* phero = new double[pheromone_matrix_size];
	int* best_sequence  = new int[n_object_type];

	initializePheromoneMatrix(n_object_type, phero); //Phero OK

    //alloc device variables
    cudaMalloc((void**) &d_phero, pheromone_matrix_size*sizeof(double));
    cudaMalloc((void**) &d_delta_phero, pheromone_matrix_size*sizeof(double));
    cudaMalloc((void**) &d_fitness, n_ant*sizeof(double));
    cudaMalloc((void**) &d_best_fitness, sizeof(double));
    cudaMalloc((void**) &d_probabilities, n_ant * n_object_type * sizeof(double));
    cudaMalloc((void**) &d_bpp_items, n_object_type *sizeof(double));
    cudaMalloc((void**) &d_bpp_items_quantity, n_object_type *sizeof(int));
    cudaMalloc((void**) &d_bpp_items_quantity_copy, n_object_type * n_ant *sizeof(int));

	cudaMalloc((void**) &d_bin_capacity, sizeof(double));
    cudaMalloc((void**) &d_sum, n_ant*sizeof(double));
	cudaMalloc((void**) &d_eta, n_ant*n_object_type*sizeof(double));
	cudaMalloc((void**) &d_tau, n_ant*n_object_type*sizeof(double));

    cudaMalloc((void**) &d_bins, n_ant* n_object_total*sizeof(int));
    cudaMalloc((void**) &d_n_ants, sizeof(int));
    cudaMalloc((void**) &d_n_objects_type, sizeof(double));
    cudaMalloc((void**) &d_n_objects_total, sizeof(double));
    cudaMalloc((void**) &d_best_solution, n_object_type*sizeof(int));

    //setup random generators TODO
    setup_rand_kernel<<<n_ant, 1, 0, stream[0]>>>(d_rand_states_ind, time(NULL));
//    checkError(0);

	double* best_fitness = (double*)malloc(1*sizeof(double));
	best_fitness[0] = 9999.9;

	cudaMemcpy(d_best_fitness, best_fitness, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n_objects_type, n_objects_types, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n_objects_total, n_objects_total, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bin_capacity, bin_capacity, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n_ants, &n_ant, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_phero, phero, pheromone_matrix_size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bpp_items, bpp_items, n_object_type*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bpp_items_quantity, bpp_qty, n_object_type*sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	int iteration = 0;

	auto t_start = std::chrono::high_resolution_clock::now();

	//START iterations
	for(iteration = 0; iteration < n_iterations; iteration++){

		//Create Solution / Start Packing
		packing_kernel<<<n_blocks, n_threads>>>(d_n_objects_type, d_n_objects_total, d_bin_capacity, d_n_ants,
				d_phero, d_bpp_items,d_bpp_items_quantity, d_bpp_items_quantity_copy, d_bins, d_eta, d_tau, d_probabilities,d_fitness, d_rand_states_ind);
//		cudaDeviceSynchronize();
		//checkError(1);

		//Pheromone Evaporation
		evaporation_kernel<<<n_objects_types[0], n_objects_types[0]>>>(d_phero);
//		cudaDeviceSynchronize();
		//checkError(2);

		//Pheromone Deposit
		update_pheromones_kernel<<<n_blocks, n_threads>>>(d_n_objects_total, d_n_objects_type, d_bin_capacity, d_phero, d_bpp_items, d_bins, d_fitness);
//		cudaDeviceSynchronize();
		//cudaDeviceSynchronize();
		//checkError(3);

		//Update Best Fitness
		//update_best_fitness_kernel<<<n_threads, n_blocks>>>(d_n_objects_total, d_bins, d_fitness, d_best_fitness);
//		optimized_update_best_fitness_kernel<<<n_threads, n_blocks, n_ant*sizeof(double)>>>(d_fitness, d_best_fitness);
		seq_update_best_fitness_kernel<<<1, 1>>>(d_n_ants, d_n_objects_total, d_bins, d_fitness, d_best_fitness);
//		cudaDeviceSynchronize();
		//cudaDeviceSynchronize();
		//checkError(4);
//		cudaDeviceSynchronize();
	}

	cudaDeviceSynchronize();
	auto t_end = std::chrono::high_resolution_clock::now();

	double time = std::chrono::duration<double>(t_end-t_start).count();

	double bf = 0.0;
	cudaMemcpy(&bf, d_best_fitness, sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("%i, %i, %i, %f , %f \n", problem_id, BLOCK_SIZE, n_ant, time, bf);

//	printf("\n Build \t evapore \t deposit \t check best \n");
//	printf("\n %f \t %f \t %f \t  %f \n\n", time1,time2,time3,time4);
//	checkError(5);

	cudaFree(d_rand_states_ind);
	cudaFree(d_phero);
	cudaFree(d_delta_phero);
	cudaFree(d_fitness);
	cudaFree(d_best_fitness);
	cudaFree(d_probabilities);
	cudaFree(d_bpp_items);
	cudaFree(d_bpp_items_quantity);
	cudaFree(d_bpp_items_quantity_copy);
	cudaFree(d_bin_capacity);
	cudaFree(d_sum);
	cudaFree(d_eta);
	cudaFree(d_tau);
	cudaFree(d_bins);
	cudaFree(d_n_ants);
	cudaFree(d_n_objects_type);
	cudaFree(d_n_objects_total);
	cudaFree(d_best_solution);

	checkError(6);

	free(n_objects_types);
	free(n_objects_total);
	free(bin_capacity);
	free(bpp_items);
	free(best_fitness);

	return bf;
}
//END OF MAIN

void checkError(int n){
	// check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
	// print the CUDA error message and exit
	printf("%i - CUDA error: %s\n",n, cudaGetErrorString(error));
	exit(-1);
  }
}


// ===  AUX FUNCTIONS  =================================================================
// =====================================================================================
// ===  FUNCTION  ======================================================================
//         Name:  ATOMIC ADD
//         Description:
// =====================================================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
