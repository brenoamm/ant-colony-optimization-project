#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include "mknap_aco_cuda_algorithm.cuh"

#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 1
#define block_size 64
#define TAUMAX 2

__device__ int d_n_objects;
__device__ int d_n_constraints;
__device__ double d_best_fitness;

using namespace std;

// ===  HOST FUNCTIONS  ================================================================
// =====================================================================================
// ===  FUNCTION  ======================================================================
//         Name:  Run ACO (Start POint)
// =====================================================================================
double run_aco(int n_ants, int n_iterations, int problem_id){

	// =======================Initialize GPUs==============================================================
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

		cudaError err = cudaGetLastError();
		if ( cudaSuccess != err )
		{
			printf("\n cudaCheckError() failed at : %s \n", cudaGetErrorString( err ) );
		}
	}

	int n_blocks = n_ants / block_size;

	// =======================Start Variables and Read Input Files==========================================
	int n_objects = 0;
	int n_constraints = 0;
	double Q = 0.0;

	setProblemVariables(problem_id, n_objects, n_constraints);

	double* object_values =  new double[n_objects];
	double* dimension_values =  new double[n_objects * n_constraints];
	double* constraint_max_values =  new double[n_objects * n_constraints];

	readInputFile(problem_id, n_objects, n_constraints, object_values, dimension_values, constraint_max_values, Q);

	// =======================Start Host Variables a========================================================
	int* best_sequence  = new int[n_objects];
	double bestFitness = 0.0; //Maximization Problem

	// =======================Start Device Variables a======================================================
	int* d_ant_solutions;
	int* d_ant_available_objects;
	int* d_best_solution;

	double* d_object_values;
	double* d_dimension_values;
	double* d_constraint_max_values;
	double* d_pheromones;
	double* d_delta_phero;
	double* d_probabilities;
	double* d_free_space; //free space for each constraint dimension (for each ant)
	double*  d_eta;
	double*  d_tau;
	double* d_ant_fitness;

	//Calculate structures' sizes
	size_t size_ant_solutions = n_ants * n_objects * sizeof(int);
	size_t size_best_solution = n_objects * sizeof(int);
	size_t size_pheromones = n_objects * n_objects * sizeof(double);
	size_t size_probabilities = n_ants * n_objects * sizeof(double);
	size_t size_objectvalues = n_objects * sizeof(double);
	size_t size_dimension_values = n_objects * n_constraints * sizeof(double);
	size_t size_constraint_max_values = n_constraints * sizeof(double);
	size_t size_free_space = n_ants * n_constraints * sizeof(double);
	size_t size_ant_fitness = n_ants * sizeof(double);

	//Aloccate Structures in GPU
	cudaMalloc(&d_ant_solutions, size_ant_solutions);
	cudaMalloc(&d_ant_available_objects, size_ant_solutions);
	cudaMalloc(&d_best_solution, size_best_solution);
	cudaMalloc(&d_pheromones, size_pheromones);
	cudaMalloc(&d_delta_phero, size_pheromones);
	cudaMalloc(&d_probabilities, size_probabilities);
	cudaMalloc(&d_object_values, size_objectvalues);
	cudaMalloc(&d_dimension_values, size_dimension_values);
	cudaMalloc(&d_constraint_max_values, size_constraint_max_values);
	cudaMalloc(&d_free_space, size_free_space);
	cudaMalloc(&d_eta, size_probabilities);
	cudaMalloc(&d_tau, size_probabilities);
	cudaMalloc(&d_ant_fitness, size_ant_fitness);

	//Init Random Generators
	curandState* d_rand_states_ind;
	cudaMalloc(&d_rand_states_ind, n_ants *  sizeof(curandState));

	setup_rand_kernel<<<n_blocks, block_size, 0, stream[0]>>>(d_rand_states_ind, time(NULL));

	// ======================= Copy variables to device ===================================================
	cudaMemcpy(d_object_values, object_values, size_objectvalues, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dimension_values, dimension_values, size_dimension_values, cudaMemcpyHostToDevice);
	cudaMemcpy(d_constraint_max_values, constraint_max_values, size_constraint_max_values, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// ======================= Init Global Variables ===================================================
	// i.e. Pheromones, Delta pheromones
	init_global_variables_kernel<<<1,1, 0, stream[0]>>>(n_ants, n_objects, n_constraints, d_pheromones, d_delta_phero,
			d_rand_states_ind, d_ant_solutions, d_ant_available_objects, d_free_space, d_constraint_max_values);
	cudaDeviceSynchronize();

	// ======================= Start ACO Iterations ====================================================
	// ======================= Start ACO Iterations ====================================================
	// ======================= Start ACO Iterations ====================================================
	int iteration = 0;
	while(iteration < n_iterations){

		//Generate Solutions
		generate_solutions_kernel<<<n_blocks, block_size, 0, stream[0]>>>(d_ant_solutions, d_ant_available_objects, d_pheromones, d_probabilities,
				d_object_values, d_dimension_values, d_constraint_max_values, d_free_space, d_eta, d_tau, d_rand_states_ind, d_ant_fitness);
		cudaDeviceSynchronize();

		//Check for best Fitness
		seq_update_best_fitness_kernel<<<1, 1, 0, stream[0]>>>(n_ants, d_ant_solutions, d_ant_fitness, d_best_solution);

		//Pheromone evaporation
		evaporation_kernel<<<n_objects,n_objects,0,stream[0]>>>(d_pheromones);

		//Pheromone deposit
		pheromone_deposit_kernel<<<n_blocks, block_size,0,stream[0]>>>(Q, d_ant_solutions, d_ant_fitness, d_pheromones, d_object_values);
		cudaDeviceSynchronize();

		iteration++;
	}

	cudaMemcpyFromSymbol(&bestFitness, d_best_fitness, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return bestFitness;
}

// ===  FUNCTION  ======================================================================
//         Name:  SetProblemVariables
//         Function:  Set Problem related variables to avoid reading inoput sfile twice
// =====================================================================================
void setProblemVariables(int problem, int& n_objects, int& n_constraints){
	switch (problem) {
		case 1:
			n_objects = 6;
			n_constraints = 10;
			break;
		case 2:
			n_objects = 10;
			n_constraints = 10;
			break;
		case 3:
			n_objects = 15;
			n_constraints = 10;
			break;
		case 4:
			n_objects = 20;
			n_constraints = 10;
			break;
		case 5:
			n_objects = 28;
			n_constraints = 10;
			break;
		case 6:
			n_objects = 39;
			n_constraints = 5;
			break;
		case 7:
			n_objects = 50;
			n_constraints = 5;
			break;
	}
}

// ===  FUNCTION  ======================================================================
//         Name:  readInputFile
//         Function:  read input file and fill data matrix
// =====================================================================================
void readInputFile(int problem_id, int n_objects, int n_constraints, double* object_values,
		double* dimension_values, double* constraint_max_values, double& Q){

		std::string file_name = "mknap1";

		switch (problem_id) {
			case 1:
				file_name = "mknap1";
				break;
			case 2:
				file_name = "mknap2";
				break;
			case 3:
				file_name = "mknap3";
				break;
			case 4:
				file_name = "mknap4";
				break;
			case 5:
				file_name = "mknap5";
				break;
			case 6:
				file_name = "mknap6";
				break;
			case 7:
				file_name = "mknap7";
				break;
		}

		ifstream inputFile(file_name);
		string sline;

		//header line ---- this line was already set manualy and therefore here ignored
		getline(inputFile, sline, '\n');
		std::istringstream linestream(sline);

		//Get Object values
		getline(inputFile, sline, '\n');
		std::istringstream linestream1(sline);
		for(int i = 0 ; i < n_objects ; i++){
			linestream1 >> object_values[i];
			Q += object_values[i];
		}

		//Get Constraint Values
		for(int i = 0 ; i < n_constraints ; i++){
			getline(inputFile, sline, '\n');
			std::istringstream linestream2(sline);
			for(int j = 0 ; j < n_objects ; j++){
				linestream2 >> dimension_values[i*n_objects +j];
			}
		}

		getline(inputFile, sline, '\n');
		std::istringstream linestream3(sline);
		for(int i = 0 ; i < n_constraints ; i++){
			linestream3 >> constraint_max_values[i];
		}

		inputFile.close();

		Q = 1/Q;
}

//void readInputFile(int problem_id, int n_objects, int n_constraints, double* object_values,
//		double* dimension_values, double* constraint_max_values, double& Q){
//
//		std::string file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap1";
//
//		switch (problem_id) {
//			case 1:
//				file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap1";
//				break;
//			case 2:
//				file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap2";
//				break;
//			case 3:
//				file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap3";
//				break;
//			case 4:
//				file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap4";
//				break;
//			case 5:
//				file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap5";
//				break;
//			case 6:
//				file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap6";
//				break;
//			case 7:
//				file_name = "/home/b/b_mene01/aco-multidimensional-knapsack-problem/build/release/mknap7";
//				break;
//		}
//
//		ifstream inputFile(file_name);
//		string sline;
//
//		//header line ---- this line was already set manualy and therefore here ignored
//		getline(inputFile, sline, '\n');
//		std::istringstream linestream(sline);
//
//		//Get Object values
//		getline(inputFile, sline, '\n');
//		std::istringstream linestream1(sline);
//		for(int i = 0 ; i < n_objects ; i++){
//			linestream1 >> object_values[i];
//			Q += object_values[i];
//		}
//
//		//Get Constraint Values
//		for(int i = 0 ; i < n_constraints ; i++){
//			getline(inputFile, sline, '\n');
//			std::istringstream linestream2(sline);
//			for(int j = 0 ; j < n_objects ; j++){
//				linestream2 >> dimension_values[i*n_objects +j];
//			}
//		}
//
//		getline(inputFile, sline, '\n');
//		std::istringstream linestream3(sline);
//		for(int i = 0 ; i < n_constraints ; i++){
//			linestream3 >> constraint_max_values[i];
//		}
//
//		inputFile.close();
//
//		Q = 1/Q;
//}

// ===  CUDA       ======================================================================
// ===  ATOMIC ADD  ======================================================================
// ===  DEFINITION  ======================================================================

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

// ===  GPU       ======================================================================
// ===  FUNCTIONs  ======================================================================
// ===  FUNCTIONs  ======================================================================
// ===  FUNCTIONs  ======================================================================
// ===  FUNCTIONs  ======================================================================
//         Name:  setup_rand_kernel
//         Function:  start random generators with seeds
// =====================================================================================
__global__ void setup_rand_kernel(curandState * state, unsigned long seed) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(seed, id, 0, &state[id]);

	__syncthreads();
}

// ===  FUNCTION  ======================================================================
//         Name:  init_global_variables_kernel
// =====================================================================================
__global__ void init_global_variables_kernel(int n_ants, int n_objects, int n_constraints, double* d_pheromones, double* d_delta_phero,
		curandState* d_rand_states_ind, int* d_ant_solutions, int* d_ant_available_objects, double* d_free_space, double* d_constraint_max_values){

	d_n_objects = n_objects;
	d_n_constraints = n_constraints;
	d_best_fitness = 0.0; //Maximization problem with positive values

	double randn = 0.0;

	for(int i = 0 ; i < n_objects * n_objects; i++){
		randn = curand_uniform(&d_rand_states_ind[1]);
		d_pheromones[i] = randn * TAUMAX;
		d_delta_phero[i] = 0.0;
	}

	for(int i = 0 ; i < n_objects * n_ants; i++){
		d_ant_solutions[i] = -1;
		d_ant_available_objects[i] = 1;
	}

	for(int i = 0 ; i < n_ants; i++){
		for(int j = 0 ; j < n_constraints; j++){
			d_free_space[i*n_constraints +j] = d_constraint_max_values[j];
		}
	}
}

// ===  FUNCTION  ======================================================================
//         Name:  generate_solutions_kernel
//         One Ant per Thread
// =====================================================================================
__global__ void generate_solutions_kernel(int* d_ant_solutions, int* d_ant_available_objects, double* d_pheromones, double* d_probabilities,
		double* d_object_values, double* d_dimension_values, double* d_constraint_max_values, double* d_free_space,  double*  d_eta, double*  d_tau,
		curandState* d_rand_states_ind, double* d_ant_fitness){

	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;
	//int solution_index = ant_index * d_n_objects;

	//Solution related Variables
	double value_object_j = 0.0;
	double pheromone_to_object_j = 0.0;
	double size_i_object_j = 0.0;
	double average_tightness_object_j = 0.0;
	double free_space_i = 0.0;
	double eta = 0.0;
	double tau = 0.0;
	double eta_tau_sum = 0.0;
	double fitness = 0.0;

	bool is_too_big = false;
	bool is_possible = false;

	//iteration to add objects
	//The maximum size of a solution is the number of objects
	//The solution stops iterating in the case no objects fit anymore.
	for(int step = 0 ; step < d_n_objects ; step++){

		eta_tau_sum = 0.0;
		is_possible = false;

		//Iterate over objects to calculate probability of chosing it as next
		for(int object_j = 0 ; object_j < d_n_objects ; object_j++){
			//Check if objects are available
			if(d_ant_available_objects[ant_index * d_n_objects + object_j] == 1){

				value_object_j = d_object_values[object_j];
				pheromone_to_object_j = d_pheromones[step * d_n_objects + object_j];

				//Calculate average tightness -> Equation 4
				average_tightness_object_j = 0.0;
				is_too_big = false;

				for(int i = 0; i < d_n_constraints ; i++){
					size_i_object_j = d_dimension_values[i*d_n_objects + object_j];
					free_space_i = d_free_space[ant_index*d_n_constraints + i];

					if(size_i_object_j <= free_space_i){
						if(free_space_i == 0.0){
							average_tightness_object_j += 1.0;
						}else{
							average_tightness_object_j += (size_i_object_j / free_space_i);
						}
					}else{
						//Object is to big and probability shall be 0
						is_too_big = true;
					}
				}

				if(!is_too_big){
					average_tightness_object_j = average_tightness_object_j / d_n_constraints;

					eta = pow((value_object_j / average_tightness_object_j), BETA);
					tau = pow(pheromone_to_object_j, ALPHA);

					eta_tau_sum += (eta * tau);

					d_eta[ant_index*d_n_objects + object_j] = eta;
					d_tau[ant_index*d_n_objects + object_j] = tau;

					is_possible = true;

				}else{ //Don't Fit -> Probability = 0;
					d_eta[ant_index*d_n_objects + object_j] = 0.0;
					d_tau[ant_index*d_n_objects + object_j] = 0.0;
				}
			}else{ //Not available -> Probability = 0;
				d_eta[ant_index*d_n_objects + object_j] = 0.0;
				d_tau[ant_index*d_n_objects + object_j] = 0.0;
			}
		}

		if(is_possible){
			//Finish Probability calculations using eta and tau
			for(int object_j = 0 ; object_j < d_n_objects ; object_j++){
				d_probabilities[ant_index * d_n_objects + object_j] =
						(d_eta[ant_index*d_n_objects + object_j] * d_tau[ant_index*d_n_objects + object_j]) / eta_tau_sum;
			}

			//Add new object in a probabilistic manner
			double random = curand_uniform(&d_rand_states_ind[ant_index]);
			int select_index = 0;
			int selected_object = 0;
			double sum = 0.0;
			double prob = 0.0;

			while ((sum <= random) && (select_index < d_n_objects)){

				prob = d_probabilities[ant_index*d_n_objects+select_index];
				if(prob > 0.0){
					sum += prob;
					selected_object = select_index;
				}

				select_index++;
			}

			d_ant_solutions[ant_index*d_n_objects + step] = selected_object;
			d_ant_available_objects[ant_index*d_n_objects + selected_object] = 0;

			for(int j = 0 ; j < d_n_constraints ; j++){
				d_free_space[ant_index*d_n_constraints+j] -= d_dimension_values[j*d_n_objects + selected_object];
			}

			fitness += d_object_values[selected_object];
		}else{
			d_ant_solutions[ant_index*d_n_objects + step] = -1;
		}
	}

	d_ant_fitness[ant_index] = fitness;

	//Reset Free Spaces
	for(int j = 0 ; j < d_n_constraints; j++){
		d_free_space[ant_index* d_n_constraints + j] = d_constraint_max_values[j];
	}
	for(int j = 0 ; j < d_n_objects; j++){
		d_ant_available_objects[ant_index* d_n_objects + j] = 1;
	}
}

// ===  FUNCTION  ======================================================================
//         Name:  seq_update_best_fitness_kernel
//         To be Optimized
// =====================================================================================
__global__ void seq_update_best_fitness_kernel(int n_ants, int* d_ant_solution, double* d_ants_fitness, int* d_best_solution){

	for(int i = 0 ; i < n_ants ; i++){
		double ant_j_fitness = d_ants_fitness[i];

		if(ant_j_fitness > d_best_fitness){
			d_best_fitness = ant_j_fitness;

			for(int j = 0 ; j<d_n_objects; j++){
				d_best_solution[j] = d_ant_solution[i*d_n_objects + j];
			}
		}
	}
}

// ===  FUNCTION  ======================================================================
//         Name:  evaporation_kernel
//         Description:
// =====================================================================================
__global__ void evaporation_kernel(double* d_phero) {

	int x_index = blockIdx.x * blockDim.x + threadIdx.x;

	//Evaporation Rate
	double RO = EVAPORATION;

	d_phero[x_index] = (1 - RO) * d_phero[x_index];
}

// ===  FUNCTION  ======================================================================
//         Name:  evaporation_kernel
//         Description: Q is a constant equals to the sum of all object values.
// =====================================================================================
__global__ void pheromone_deposit_kernel(double Q, int* d_ant_solutions, double* d_ant_fitness, double* d_pheromones, double* d_object_values){

	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;
	int object_i = 0;
	double delta_phero = 0.0;
	double value = 0.0;

	for(int i = 0 ; i < d_n_objects ; i++){

		object_i = d_ant_solutions[ant_index * d_n_objects + i];

		if(object_i != -1){
			value = d_object_values[object_i];
			delta_phero = Q * value;
			atomicAdd(&d_pheromones[i*d_n_objects + object_i],  delta_phero);
		}

		//Restart Solutions
		d_ant_solutions[ant_index * d_n_objects + i] = -1;
	}
}
