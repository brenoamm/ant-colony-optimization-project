#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

// ===  HOST FUNCTIONS  ================================================================
// =====================================================================================
// ===  FUNCTION  ======================================================================
//         Name:  Run ACO (Start POint)
// =====================================================================================
double run_aco(int n_ants, int n_iterations, int problem_id);

// ===  FUNCTION  ======================================================================
//         Name:  SetProblemVariables
//         Function:  Set Problem related variables to avoid reading inoput sfile twice
// =====================================================================================
void setProblemVariables(int problem, int& n_objects, int& n_constraints);

// ===  FUNCTION  ======================================================================
//         Name:  readInputFile
//         Function:  read input file and fill data matrix
// =====================================================================================
void readInputFile(int problem_id, int n_objects, int n_constraints, double* object_values,
		double* dimension_values, double* constraint_max_values, double& Q);


// ===  GPU FUNCTIONS  =================================================================
// =====================================================================================
// ===  FUNCTION  ======================================================================
//         Name:  setup_rand_kernel
// =====================================================================================
__global__ void setup_rand_kernel(curandState * state, unsigned long seed);

// ===  FUNCTION  ======================================================================
//         Name:  init_global_variables_kernel
// =====================================================================================
__global__ void init_global_variables_kernel(int n_ants, int n_objects, int n_constraints, double* d_pheromones,
		double* d_delta_phero, curandState* d_rand_states_ind, int* d_ant_solutions, int* d_ant_available_objects,
		double* d_free_space, double* d_constraint_max_values);

// ===  FUNCTION  ======================================================================
//         Name:  generate_solutions_kernel (route/pack)
// =====================================================================================
__global__ void generate_solutions_kernel(int* d_ant_solutions, int* d_ant_available_objects, double* d_pheromones, double* d_probabilities,
		double* d_object_values, double* d_dimension_values, double* d_constraint_max_values, double* d_free_space, double*  d_eta, double*  d_tau,
		curandState* d_rand_states_ind, double* d_ant_fitness);

// ===  FUNCTION  ======================================================================
//         Name:  seq_update_best_fitness_kernel (route/pack)
// =====================================================================================
__global__ void seq_update_best_fitness_kernel(int n_ants, int* d_ant_solution, double* d_ants_fitness, int* d_best_solution);

// ===  FUNCTION  ======================================================================
//         Name:  evaporation_kernel
// =====================================================================================
__global__ void evaporation_kernel(double* d_phero);

// ===  FUNCTION  ======================================================================
//         Name:  pheromone_deposit_kerne
// =====================================================================================
__global__ void pheromone_deposit_kernel(double Q, int* d_ant_solutions, double* d_ant_fitness, double* d_pheromones, double* d_object_values);
