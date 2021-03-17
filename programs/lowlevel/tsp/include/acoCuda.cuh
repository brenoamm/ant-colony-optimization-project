

void readMap(double* coord, double* phero, double* dist, int NCITY);

void pheroEvap(double* phero, int NCITY);

double pheroDeposit(int* sequence, int formiga, double* dist, double* phero, int NCITY);

double length (int antk, int NUMBEROFCITIES, int* ROUTES, double* DIST);

void updatePHEROMONES (int NUMBEROFANTS, int NUMBEROFCITIES, int* ROUTES, double* PHEROMONES, double* DIST, double bestRoute);

__global__ void setup_rand_kernel(curandState * state, unsigned long seed);

__global__ void calcDist(double* dist, double* coord, int NCITY);

__global__ void route(double* phero, double* dist, int* sequence, int NCITY);

__global__ void route2(double* PHERO, double* DIST, int* ROUTES, double* PROBS, int NCITY, curandState* rand_states);

__global__ void route3(double* PHERO, double* DIST, int* ROUTES, int NCITY, curandState* rand_states);

__global__ void d_updatePHEROMONES (int* NUMBEROFANTS, int* NUMBEROFCITIES, int* ROUTES, double* PHEROMONES, double* DELTAPHEROMONES, double* DIST, double* routes_distance, double* bestRoute, int* d_best_sequence);

__device__ double d_length (int antk, int NUMBEROFCITIES, int* ROUTES, double* DIST);

__device__ bool vizited(int antk, int c, int* ROUTES, int NUMBEROFCITIES, int i);

__device__ double PHI (int cityi, int cityj, int antk, int NUMBEROFCITIES ,double* dist, double* PHEROMONES, int* ROUTES, int step);

__device__ int city (int antK, int NCITIES, double* PROBS, curandState* rand_states);

